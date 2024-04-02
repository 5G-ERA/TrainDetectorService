import argparse
import logging
import numpy as np
import os
import time
import traceback

from functools import partial
from queue import Queue
from typing import Dict, Optional, Tuple, Any

from era_5g_train_detection_standalone.worker import TrainDetectorWorker

from era_5g_interface.channels import CallbackInfoServer, ChannelType, DATA_NAMESPACE, DATA_ERROR_EVENT
from era_5g_interface.dataclasses.control_command import ControlCommand, ControlCmdType
from era_5g_interface.interface_helpers import HeartbeatSender
from era_5g_interface.task_handler_internal_q import TaskHandlerInternalQ

from era_5g_server.server import NETAPP_STATUS_ADDRESS, NetworkApplicationServer, generate_application_heartbeat_data


logger = logging.getLogger("TrainDetectorService NetApp interface")

# port of the netapp's server
NETAPP_PORT = os.getenv("NETAPP_PORT", 5896)
# input queue size
NETAPP_INPUT_QUEUE = int(os.getenv("NETAPP_INPUT_QUEUE", 15))


class Server(NetworkApplicationServer):
    """Server receives images and commands from client, sends them to worker, and sends results to client."""

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Constructor.

        Args:
            *args: NetworkApplicationServer arguments.
            **kwargs: NetworkApplicationServer arguments.
        """

        super().__init__(
            callbacks_info={
                "image_jpeg": CallbackInfoServer(ChannelType.JPEG, self.image_callback),
                "image_h264": CallbackInfoServer(ChannelType.H264, self.image_callback),
                "json": CallbackInfoServer(ChannelType.JSON, self.json_callback),
            },
            *args,
            **kwargs,
        )

        # The detector to be used.
        self.detector_threads: Dict[str, TrainDetectorWorker] = {}

        # Dict of registered tasks.
        self.tasks: Dict[str, TaskHandlerInternalQ] = {} 

        # Create Heartbeat sender
        self.heartbeat_sender = HeartbeatSender(NETAPP_STATUS_ADDRESS, self.generate_heartbeat_data)

    def generate_heartbeat_data(self):
        """Heart beat generation and sending."""

        latencies = []

        queue_occupancy = 0
        queue_size = 0
        for task in self.tasks.values():
            queue_occupancy += task.data_queue_occupancy()
            queue_size += task.data_queue_size()
        for worker in self.detector_threads.values():
            latencies.extend(worker.latency_measurements.get_latencies())
        avg_latency = 0
        if len(latencies) > 0:
            avg_latency = float(np.mean(np.array(latencies)))

        return generate_application_heartbeat_data(avg_latency, queue_size, queue_occupancy, len(self.tasks))

    def image_callback(self, sid: str, data: Dict[str, Any]):
        """Allows to receive decoded image using the websocket transport.

        Args:
            sid (str): Namespace sid.
            data (Dict[str, Any]): Data dict including decoded frame (data["frame"]) and send timestamp
                (data["timestamp"]).
        """

        #logging.debug("Image recieved")
        recv_timestamp = time.perf_counter_ns()
        
        eio_sid = self._sio.manager.eio_sid_from_sid(sid, DATA_NAMESPACE)

        if eio_sid not in self.tasks:
            logger.error(f"Non-registered client tried to send data")
            self.send_data({"message": "Non-registered client tried to send data"}, DATA_ERROR_EVENT, sid=sid)
            return

        task = self.tasks[eio_sid]
            
        task.store_data(
            {"sid": eio_sid,
            "timestamp": data["timestamp"],
            "recv_timestamp": recv_timestamp,
            "decoded": True},  # Currently, the image is always decoded before it is received here
            data["frame"]
        )

    def json_callback(self, sid: str, data: Dict):
        """
        Allows to receive general json data using the websocket transport

        Args:
            sid (str): Namespace sid.
            data (Dict): 5G-ERA Network Application specific JSON data.
        """
        # Currently not used

        logger.info(f"Client with task id: {self.get_eio_sid_of_data(sid)} sent data {data}")

    def command_callback(self, command: ControlCommand, sid: str) -> Tuple[bool, str]:
        """Pass control command to the worker to change its internal state.

        Args:
            command (ControlCommand): Control command to be processed.
            sid (str): Namespace sid.
        
        Returns:
            (result (bool), message (str)): Result of control command processing and message with description.
        """

        eio_sid = self.get_eio_sid_of_control(sid)

        logger.debug(f"Processing control command {command}, session id: {sid}")

        if command.cmd_type == ControlCmdType.INIT: 
            # Check that initialization has not been called before
            if eio_sid in self.tasks:
                logger.error(f"Client attempted to call initialization multiple times.")
                self.send_command_error("Initialization has already been called before", sid)
                return False, "Initialization has already been called before"
            
            # queue with received images
            image_queue = Queue(NETAPP_INPUT_QUEUE)

            task = TaskHandlerInternalQ(image_queue)

            try:
                # create worker and run it as thread, listening to image_queue
                send_function = partial(self.send_data, event="results", sid=self.get_sid_of_data(eio_sid))
                worker = TrainDetectorWorker(image_queue, send_function, name=f"Detector {eio_sid}", daemon=True)
            except Exception as ex:
                logger.error(f"Failed to create Detector: {repr(ex)}")
                logger.error(traceback.format_exc())
                self.send_command_error(f"Failed to create Detector: {repr(ex)}", sid)
                return False, f"Failed to create Detector: {repr(ex)}"

            worker.start()

            self.tasks[eio_sid] = task
            self.detector_threads[eio_sid] = worker

        else:
            # check that task is running
            if eio_sid not in self.tasks:
                logger.error(f"Non-registered client tried to send control command.")
                self.send_command_error("Non-registered client tried to send control command", sid)
                return False, "Network Application not initialized"
            
            task = self.tasks[eio_sid]

            if command.clear_queue:
                task.clear_storage()

            task.store_control_data(command)
        
        return True, (
                f"Control command applied, eio_sid {eio_sid}, sid {sid}, results sid"
                f" {self.get_sid_of_data(eio_sid)}, command {command}"
        )
    
    def disconnect_callback(self, sid: str) -> None:
        """Called when client disconnects - deletes task and worker.

        Args:
            sid (str): Namespace sid.
        """

        eio_sid = self.get_eio_sid_of_data(sid)
        self.tasks.pop(eio_sid)
        detector = self.detector_threads.pop(eio_sid)
        detector.stop()
        logger.info(f"Client disconnected from {DATA_NAMESPACE} namespace: session id: {sid}")


def main():
    logging.getLogger().setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(description="Train Detector Service NetApp")
    parser.add_argument(
        "--detector",
        default="mmdetection",
        help="This argument is currently ignored."
    )
    parser.add_argument("-m", "--measuring", type=bool, help="Enable extended measuring logs", default=False)
    args = parser.parse_args()
        
    logger.info("Starting Train Detector Service interface.")
    logger.info(f"The size of the queue set to: {NETAPP_INPUT_QUEUE}")

    # runs the flask server
    server = Server(port=NETAPP_PORT, host="0.0.0.0", extended_measuring=args.measuring)

    try:
        server.run_server()
    except KeyboardInterrupt:
        logger.info("Terminating ...")


if __name__ == '__main__':
    main()
