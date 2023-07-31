
from queue import Empty, Queue
from threading import Thread, Event
import logging
import cv2

import numpy as np
import time

from era_5g_interface.dataclasses.control_command import ControlCommand, ControlCmdType
from era_5g_interface.interface_helpers import LatencyMeasurements

from era_5g_train_detection_standalone.modules.mm_detector import MMDetector
from era_5g_train_detection_standalone.modules.detection_processing import DetectionProcessing


# Print processing times and FPS
DEBUG_SHOW_FPS = False

# Show processed images and detections 
DEBUG_DRAW_DETECTIONS = False


class TrainDetectorWorker(Thread):
    """Worker object for Train Detector Service.

    Worker reads data from passed queue, performs object detection and
    further processing to determine if objects are moving and returns 
    the results.
    """

    def __init__(self, image_queue: Queue, sio, sort_tracker_params=None, optical_flow_params= None, **kw):
        """
        Constructor

        Args:
            image_queue (Queue): The queue with all to-be-processed images.
            app (_type_): The flask app for results publishing.
            sort_tracker_params (dict, optional): Dictionary with 
                optional SORT tracker parameters.
            optical_flow_params (dict, optional): Dictionary with 
                optional parameters for optical flow calculation.
        """

        super().__init__(**kw)
        self.image_queue = image_queue
        self.sio = sio

        self.stop_event = Event()

        self.time = None 
        self.fps = 0.0 

        self.latency_measurements = LatencyMeasurements()

        # Variables for measuring processing speed
        self.proc_times_len = 10  # Number of measuements for averaging
        self.processing_times = np.zeros(self.proc_times_len)
        self.ring_buffer_index = 0

        # Init detector
        train_class_id = 6  # id of object class "train"
        self.detector = MMDetector(class_id_filter=train_class_id)

        # Create object for processing of detections
        self.detection_processing = DetectionProcessing(sort_tracker_params, optical_flow_params, 
                                                        render_detections=DEBUG_DRAW_DETECTIONS)

    def stop(self):
        self.stop_event.set()

    def run(self):
        """
        Periodically reads images from python internal queue processes them.
        """

        logging.debug(f"TrainDetectorWorker thread is running.")

        while not self.stop_event.is_set():
            # Get image and metadata from input queue
            try:
                queue_data = self.image_queue.get(block=True, timeout=1)
                
                # Check for ControlCommand token (enables clearing of internal state)
                if isinstance(queue_data, ControlCommand):
                    if queue_data.cmd_type == ControlCmdType.RESET_STATE:
                        self.detection_processing.clear_tracks()
                        logging.debug("Internal state cleared.")
                    else:
                        logging.warning(f"Got control command with type {queue_data.cmd_type}, which is not applicable.")
                    continue

                metadata, image = queue_data
            except Empty:
                continue

            if not metadata.get("decoded", True):
                # Decode image
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
            # Prepare dictionary with all data used during processing
            data = {"image": image}

            if DEBUG_SHOW_FPS:
                t1 = time.time() # Start timer

            metadata["timestamp_before_process"] = time.perf_counter_ns()

            # Detect individual obejcts
            detections = self.detector.process_image(image)
            data.update(detections)  # Add detections to the dict with data

            # Process detections and determine if there is any movement
            self.detection_processing.process_detections(data)

            metadata["timestamp_after_process"] = time.perf_counter_ns()
            
            if DEBUG_SHOW_FPS:
                self.processing_times[self.ring_buffer_index] = time.time() - t1
                self.ring_buffer_index = (self.ring_buffer_index + 1) % self.proc_times_len
                avg_proc_time = np.mean(self.processing_times)
                fps = 1.0 / avg_proc_time
                logging.debug(f"avg_proc_time: {avg_proc_time}, fps: {fps}")

            if DEBUG_DRAW_DETECTIONS:
                # Draw the detections and movements 
                out_img = data["output_bgr_image"]
                window_name = f"Detections {self.ident}"
                cv2.imshow(window_name, out_img)
                cv2.waitKey(1)

            self.publish_results(data, metadata)  

    def publish_results(self, data, metadata):
        """Publishes the results to the robot.

        Args:
            data (_type_): The results of the motion estimation.
            metadata (_type_): NetApp-specific metadata related to processed image.

        """

        send_timestamp = time.perf_counter_ns()

        self.latency_measurements.store_latency(send_timestamp - metadata["recv_timestamp"])

        # Send only the necessary data about movements (no images, etc.),
        # add timestamp to the results
        r = {"timestamp": metadata["timestamp"],
             "recv_timestamp": metadata["recv_timestamp"],
             "timestamp_before_process": metadata["timestamp_before_process"],
             "timestamp_after_process": metadata["timestamp_after_process"],
             "send_timestamp": send_timestamp,
             "movements": data["movements"], # list(dict("bbox": bbox, "moving": 0/1/-1))
                # bbox is: x1, y1, x2, y2 (top-left bottom-right corners)
            }

        self.sio.emit("message", r, namespace='/results', to=metadata["websocket_id"])


