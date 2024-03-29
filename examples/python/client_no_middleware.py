from __future__ import annotations

import argparse
import logging
import os
import signal
import time
import traceback

from queue import Queue
from types import FrameType
from typing import Any, Dict, Optional

import cv2
import numpy as np

from era_5g_client.client_base import NetAppClientBase
from era_5g_client.exceptions import FailedToConnect
from era_5g_interface.channels import CallbackInfoClient, ChannelType
from era_5g_interface.utils.rate_timer import RateTimer
from era_5g_interface.dataclasses.control_command import ControlCmdType, ControlCommand

from utils.results_viewer import ResultsViewer


image_storage: Dict[int, np.ndarray] = dict()
results_storage: Queue[Dict[str, Any]] = Queue()
stopped = False
verbose = False

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


def get_results(results: Dict[str, Any]) -> None:
    """Callback which process the results from the NetApp.

    Args:
        results (str): The results in json format
    """

    if verbose:
        print(results)

    if "timestamp" not in results:
        return

    if results_storage is not None:
        results_storage.put(results, block=False)
 

def main() -> None:
    """Creates the client class and starts the data transfer."""

    parser = argparse.ArgumentParser(description='Example client communication without middleware.')
    parser.add_argument("-n", "--no-results",
        default=False, action="store_true",
        help="Do not show window with visualization of detection results. Defaults to False."
        )
    parser.add_argument("-v", "--verbose",
        default=False, action="store_true",
        help="Print information about processed data. Defaults to False."
        )
    parser.add_argument("-c", "--crop",
        default=False, action="store_true",
        help="Crop fisheye image from BringAuto."
        )
    parser.add_argument("--h264",
        default=False, action="store_true",
        help="Use h264 compression."
        )
    args = parser.parse_args()
    global verbose
    verbose = args.verbose

    logging.getLogger().setLevel(logging.INFO)

    global results_storage
    if args.no_results:
        results_storage = None
    else:
        results_viewer = ResultsViewer(image_storage, results_storage, name="test_client_http_viewer", daemon=True)
        results_viewer.start()

    client = None
    global stopped
    stopped = False

    def signal_handler(sig: int, frame: Optional[FrameType]) -> None:
        global stopped
        stopped = True
        if not args.no_results:
            results_viewer.stop()
        print(f"Terminating ({signal.Signals(sig).name})...")

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        if FROM_SOURCE:
            # creates a video capture to pass images to the 5G-ERA Network Application either from webcam ...
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Cannot open camera")
        else:
            # or from video file
            cap = cv2.VideoCapture(TEST_VIDEO_FILE)
            if not cap.isOpened():
                raise Exception("Cannot open video file")
            
        fps = cap.get(cv2.CAP_PROP_FPS)

        #cap.set(cv2.CAP_PROP_POS_MSEC, 9000)  # debug 

        # creates an instance of NetApp client with results callback
        client = NetAppClientBase({"results": CallbackInfoClient(ChannelType.JSON, get_results)})
        # register with an ad-hoc deployed NetApp
        netapp_address = f"http://{NETAPP_ADDRESS}:{NETAPP_PORT}/"
        target_w, target_h = 640, 480
        channel_name = "image_jpeg"
        channel_type = ChannelType.JPEG
        if args.h264:
            channel_name = "image_h264"
            channel_type = ChannelType.H264
        client.register(netapp_address)

        # create timer to ensure required fps speed of the sending loop
        logging.info(f"Using RateTimer with {fps} FPS.")
        rate_timer = RateTimer(rate=fps, iteration_miss_warning=True)

        while not stopped:
            ret, frame = cap.read()
            timestamp = time.perf_counter_ns()
            if not ret:
                break

            if args.crop:
                # Used for testing with raw fisheye images from BringAuto
                #print(frame.shape) 720, 1280, 3
                frame = frame[int(frame.shape[0]/4):int(frame.shape[0]*3/4), int(frame.shape[1]/2):, :]
                resized = frame
            else:
                resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            
            if not args.no_results:
                image_storage[timestamp] = resized

            rate_timer.sleep()  # sleep until next frame should be sent (with given fps)
            client.send_image(resized, channel_name, channel_type, timestamp)

        # Send command to reset internal state of the NetApp
        control_cmd = ControlCommand(ControlCmdType.RESET_STATE, clear_queue=True)
        client.send_control_command(control_cmd)
        time.sleep(2)

    except FailedToConnect as ex:
        print(f"Failed to connect to server ({ex})")
    except KeyboardInterrupt:
        print("Terminating...")
    except Exception as ex:
        traceback.print_exc()
        print(f"Failed to create client instance ({ex})")
    finally:
        if client is not None:
            client.disconnect()


if __name__ == "__main__":
    main()
