
import cv2
import logging
import time

from queue import Queue, Empty
from threading import Event, Thread

import cv2
import numpy as np


DEBUG_PRINT_DELAY = False  # prints the delay between capturing image and receiving the results


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
