from __future__ import annotations

import logging
import math
import os
import signal
import time
import traceback
from queue import Queue, Empty
from threading import Event, Thread
from types import FrameType
from typing import Any, Dict, Optional

import cv2
import numpy as np

from era_5g_client.client_base import NetAppClientBase
from era_5g_client.exceptions import FailedToConnect

from era_5g_client.dataclasses import NetAppLocation

from utils.rate_timer import RateTimer

image_storage: Dict[str, np.ndarray] = dict()
results_storage: Queue[Dict[str, Any]] = Queue()
stopped = False

DEBUG_PRINT_DELAY = False  # prints the delay between capturing image and receiving the results

# Video from source flag
FROM_SOURCE = False
# ip address or hostname of the computer, where the netapp is deployed
NETAPP_ADDRESS = os.getenv("NETAPP_ADDRESS", "127.0.0.1")
# port of the netapp's server
NETAPP_PORT = int(os.getenv("NETAPP_PORT", 5896))
# test video file
try:
    TEST_VIDEO_FILE = os.environ["TEST_VIDEO_FILE"]
except KeyError as e:
    raise Exception(f"Failed to run example, env variable {e} not set.")

if not os.path.isfile(TEST_VIDEO_FILE):
    raise Exception("TEST_VIDEO_FILE does not contain valid path to a file.")


class ResultsViewer(Thread):
    def __init__(self, image_storage, results_queue, **kw) -> None:
        super().__init__(**kw)
        self.image_storage = image_storage
        self.results_queue = results_queue
        self.stop_event = Event()
        self.index = 0

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        logging.info("Thread %s: starting", self.name)
        while not self.stop_event.is_set():
            try:
                results = self.results_queue.get(timeout=1)
            except Empty:
                continue
            timestamp_str = results["timestamp"]
            timestamp = int(timestamp_str)
            if DEBUG_PRINT_DELAY:
                time_now = time.time_ns()
                print(f"{(time_now - timestamp) * 1.0e-9:.3f}s delay")
            try:
                frame = self.image_storage.pop(timestamp_str)

                detections = results["movements"]
                for det in detections:
                    moving = det["moving"]
                    if moving == -1:
                        color = (0, 165, 255) # orange
                    elif moving == 1:
                        color = (0, 0, 255) # red
                    else:
                        color = (0, 255, 0) # green
                    thickness = 2
                    b = det["bbox"]
                    cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, thickness)

                try:
                    cv2.imshow("Results", frame)
                    cv2.waitKey(1)
                except Exception as ex:
                    print(ex)
                self.results_queue.task_done()
            except KeyError as ex:
                print(ex)


def get_results(results: Dict[str, Any]) -> None:
    """Callback which process the results from the NetApp.

    Args:
        results (str): The results in json format
    """

    print(results)
    if "timestamp" in results:
        results_storage.put(results, block=False)
    pass


def main() -> None:
    """Creates the client class and starts the data transfer."""

    results_viewer = ResultsViewer(image_storage, results_storage, name="test_client_http_viewer", daemon=True)
    results_viewer.start()

    logging.getLogger().setLevel(logging.INFO)

    client = None
    global stopped
    stopped = False

    def signal_handler(sig: int, frame: Optional[FrameType]) -> None:
        global stopped
        stopped = True
        results_viewer.stop()
        print(f"Terminating ({signal.Signals(sig).name})...")

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # creates an instance of NetApp client with results callback
        client = NetAppClientBase(get_results)
        # register with an ad-hoc deployed NetApp
        client.register(NetAppLocation(NETAPP_ADDRESS, NETAPP_PORT))
        if FROM_SOURCE:
            # creates a video capture to pass images to the NetApp either from webcam ...
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Cannot open camera")
        else:
            # or from video file
            cap = cv2.VideoCapture(TEST_VIDEO_FILE)
            if not cap.isOpened():
                raise Exception("Cannot open video file")

        # create timer to ensure required fps speed of the sending loop
        fps = cap.get(cv2.CAP_PROP_FPS)
        logging.info(f"Using RateTimer with {fps} FPS.")
        rate_timer = RateTimer(rate=fps, iteration_miss_warning=True)

        while not stopped:
            ret, frame = cap.read()
            timestamp = time.time_ns()
            if not ret:
                break
            resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            timestamp_str = str(timestamp)
            image_storage[timestamp_str] = resized

            rate_timer.sleep()  # sleep until next frame should be sent (with given fps)
            client.send_image_http(resized, timestamp_str, 5)

    except FailedToConnect as ex:
        print(f"Failed to connect to server ({ex})")
    except KeyboardInterrupt:
        print("Terminating...")
    except Exception as ex:
        traceback.print_exc()
        print(f"Failed to create client instance ({ex})")
    finally:
        results_viewer.stop()
        if client is not None:
            client.disconnect()


if __name__ == "__main__":
    main()
