
from queue import Empty, Queue
from threading import Thread, Event
import flask_socketio
import logging
import cv2

import numpy as np
import time

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
    results using the flask app. 
    """

    def __init__(self, image_queue: Queue, app, sort_tracker_params=None, optical_flow_params= None, **kw):
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
        self.app = app

        self.stop_event = Event()

        self.time = None 
        self.fps = 0.0 

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
                queue_data = self.image_queue.get(block=True)
                # TODO: check for reset token (to enable clearing of internal state)
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

            # Detect individual obejcts
            detections = self.detector.process_image(image)
            data.update(detections)  # Add detections to the dict with data

            # Process detections and determine if there is any movement
            self.detection_processing.process_detections(data)
            
            if DEBUG_SHOW_FPS:
                self.processing_times[self.ring_buffer_index] = time.time() - t1
                self.ring_buffer_index = (self.ring_buffer_index + 1) % self.proc_times_len
                avg_proc_time = np.mean(self.processing_times)
                fps = 1.0 / avg_proc_time
                logging.debug(f"avg_proc_time: {avg_proc_time}, fps: {fps}")

            if DEBUG_DRAW_DETECTIONS:
                # Draw the detections and movements 
                out_img = data["output_bgr_image"]
                cv2.imshow("Detections", out_img)
                cv2.waitKey(1)

            self.publish_results(data, metadata)  

    def publish_results(self, data, metadata):
        """Publishes the results to the robot.

        Args:
            data (_type_): The results of the motion estimation.
            metadata (_type_): NetApp-specific metadata related to processed image.

        """

        # Send only the necessary data about movements (no images, etc.),
        # add timestamp to the results
        r = {"timestamp": metadata["timestamp"],
             "movements": data["movements"], # list(dict("bbox": bbox, "moving": 0/1/-1))
                # bbox is: x1, y1, x2, y2 (top-left bottom-right corners)
            }

        # use the flask app to return the results
        with self.app.app_context():
            flask_socketio.send(r, namespace='/results', to=metadata["websocket_id"])


