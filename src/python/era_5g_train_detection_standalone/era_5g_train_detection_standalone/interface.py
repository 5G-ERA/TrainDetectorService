import base64
import argparse
import binascii
import cv2
import os
import numpy as np
import logging
import time
import socketio
import threading

from queue import Queue
from typing import Dict

from flask import Flask

from era_5g_train_detection_standalone.worker import TrainDetectorWorker
from era_5g_interface.task_handler import TaskHandler
from era_5g_interface.task_handler_internal_q import TaskHandlerInternalQ

from era_5g_interface.h264_decoder import H264Decoder
from era_5g_interface.interface_helpers import HeartBeatSender
from era_5g_interface.interface_helpers import MIDDLEWARE_REPORT_INTERVAL
from era_5g_interface.dataclasses.control_command import ControlCommand, ControlCmdType

logger = logging.getLogger("TrainDetectorService NetApp interface")

# port of the netapp's server
NETAPP_PORT = os.getenv("NETAPP_PORT", 5896)
# input queue size
NETAPP_INPUT_QUEUE = int(os.getenv("NETAPP_INPUT_QUEUE", 1))


# the max_http_buffer_size parameter defines the max size of the message to be passed
sio = socketio.Server(async_mode='threading', async_handlers=False, max_http_buffer_size=5 * (1024 ** 2))
app = Flask(__name__)
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)


detector_threads: Dict[str, TrainDetectorWorker] = {}
tasks: Dict[str, TaskHandler] = dict()


heart_beat_sender = HeartBeatSender()


def heart_beat_timer():
    latencies = []
    for worker in detector_threads.values():
        latencies.extend(worker.latency_measurements.get_latencies())
    avg_latency = 0
    if len(latencies) > 0:
        avg_latency = float(np.mean(np.array(latencies)))

    queue_size = 1
    queue_occupancy = 1

    heart_beat_sender.send_middleware_heart_beat(
        avg_latency=avg_latency, queue_size=queue_size, queue_occupancy=queue_occupancy, current_robot_count=len(tasks)
    )
    threading.Timer(MIDDLEWARE_REPORT_INTERVAL, heart_beat_timer).start()


heart_beat_timer()


def get_sid_of_namespace(eio_sid, namespace):
    return sio.manager.sid_from_eio_sid(eio_sid, namespace)

def get_results_sid(eio_sid):
    return sio.manager.sid_from_eio_sid(eio_sid, "/results")


@sio.on('connect', namespace='/data')
def connect_data(sid, environ):
    """Creates a websocket connection to the client for passing the data."""

    logger.info(f"Connected data. Session id: {sio.manager.eio_sid_from_sid(sid, '/data')}, ws_sid: {sid}")
    sio.send("You are connected", namespace='/data', to=sid)


@sio.on('connect', namespace='/control')
def connect_control(sid, environ):
    """Creates a websocket connection to the client for passing control commands."""

    logger.info(f"Connected control. Session id: {sio.manager.eio_sid_from_sid(sid, '/control')}, ws_sid: {sid}")
    sio.send("You are connected", namespace='/control', to=sid)


@sio.on('connect', namespace='/results')
def connect_results(sid, environ):
    """Creates a websocket connection to the client for passing the results."""

    logger.info(f"Connected results. Session id: {sio.manager.eio_sid_from_sid(sid, '/results')}, ws_sid: {sid}")
    sio.send("You are connected", namespace='/results', to=sid)


@sio.on('image', namespace='/data')
def image_callback_websocket(sid, data: dict):
    """
    Allows to receive jpg-encoded image using the websocket transport

    Args:
        data (dict): A base64 encoded image frame and (optionally) related timestamp in format:
            {'frame': 'base64data', 'timestamp': 'int'}

    Raises:
        ConnectionRefusedError: Raised when attempt for connection were made
            without registering first or frame was not passed in correct format.
    """
    logging.debug("A frame recieved using ws")
    recv_timestamp = time.perf_counter_ns()
    
    if 'timestamp' in data:
        timestamp = data['timestamp']
    else:
        logger.info("Timestamp not set, setting default value")
        timestamp = 0

    eio_sid = sio.manager.eio_sid_from_sid(sid, '/data')

    if eio_sid not in tasks:
        logger.error(f"Non-registered client tried to send data")
        sio.emit(
            "image_error",
            {"timestamp": timestamp,
             "error": "Not connected"},
            namespace='/data',
            to=sid
        )
        return

    if 'frame' not in data:
        logger.error(f"Data does not contain frame.")
        sio.emit(
            "image_error",
            {"timestamp": timestamp,
             "error": "Data does not contain frame."},
            namespace='/data',
            to=sid
        )
        return

    task = tasks[eio_sid]
    decoded = False

    try:
        if task.decoder:
            image = task.decoder.decode_packet_data(data["frame"])
            decoded = True
        else:
            frame = base64.b64decode(data["frame"])
            image = np.frombuffer(frame, dtype=np.uint8)
    except (ValueError, binascii.Error, Exception) as error:
        logger.error(f"Failed to decode frame data: {error}")
        sio.emit(
            "image_error",
            {"timestamp": timestamp,
             "error": f"Failed to decode frame data: {error}"},
            namespace='/data',
            to=sid
        )
    
    task.store_image(
        {"sid": eio_sid,
        "timestamp": timestamp,
        "recv_timestamp": recv_timestamp,
        "websocket_id": get_results_sid(sio.manager.eio_sid_from_sid(sid, "/data")),
        "decoded": decoded},
        image
    )


@sio.on('json', namespace='/data')
def json_callback_websocket(sid, data):
    """
    Allows to receive general json data using the websocket transport

    Args:
        data (dict): NetApp-specific json data

    Raises:
        ConnectionRefusedError: Raised when attempt for connection were made
            without registering first.
    """
    print(data)
    logger.info(f"Client with task id: {sio.manager.eio_sid_from_sid(sid, '/data')} sent data {data}")


@sio.on('command', namespace='/control') 
def control_command_callback_websocket(sid, data: Dict): 
    """Pass control command to the worker to change its internal state.

    Args:
        data (dict): Json data with the control command
    """

    eio_sid = sio.manager.eio_sid_from_sid(sid, '/control')

    logger.debug(f"Client with task id: {eio_sid} sent control data {data}")

    try:
        command = ControlCommand(**data)
    except:
        pass # Send error information to client

    if command.cmd_type == ControlCmdType.SET_STATE:  # TODO: change ControlCmdType to INIT_NETAPP

        # Check that NetApp has not been initialized before
        if eio_sid in tasks:
            logger.error(f"Client attempted to initialize the NetApp multiple times.")
            sio.emit(
                "control_cmd_error",
                {"error": "NetApp already initialized."},
                namespace='/control',
                to=sid
            )
            return

        args = command.data
        h264 = False
        if args:
            h264 = args.get("h264", False)
            fps = args.get("fps", 30)
            width = args.get("width", 0)
            height = args.get("height", 0)
            logger.info(f"H264: {h264}")
            logger.info(f"Video {width}x{height}, {fps} FPS")
        
        # queue with received images
        image_queue = Queue(NETAPP_INPUT_QUEUE)

        if h264:
            task = TaskHandlerInternalQ(eio_sid, image_queue, decoder=H264Decoder(fps, width, height))
        else:
            task = TaskHandlerInternalQ(eio_sid, image_queue)

        # create worker and run it as thread, listening to image_queue
        worker = TrainDetectorWorker(image_queue, sio, name=f"Detector {eio_sid}", daemon=True)
        worker.start()

        tasks[eio_sid] = task
        detector_threads[eio_sid] = worker

    else:
        # check that task is running
        if eio_sid not in tasks:
            logger.error(f"Non-registered client tried to send control command.")
            sio.emit(
                "control_cmd_error",
                {"error": "NetApp not initialized"},
                namespace='/control',
                to=sid
            )
            return

        task = tasks[eio_sid]

        if command.clear_queue:
            task.clear_storage()

        task.store_control_data(command)


@sio.on('disconnect', namespace='/data')
def disconnect_data(sid):
    eio_sid = sio.manager.eio_sid_from_sid(sid, "/data")
    task = tasks.pop(eio_sid)
    detector = detector_threads.pop(eio_sid)
    detector.stop()
    logger.info(f"Client disconnected from /data namespace: session id: {sid}")


@sio.on('disconnect', namespace='/control')
def disconnect_control(sid):
    logger.info(f"Client disconnected from /control namespace: session id: {sid}")


@sio.on('disconnect', namespace='/results')
def disconnect_results(sid):
    logger.info(f"Client disconnected from /results namespace: session id: {sid}")


def main():
    logging.getLogger().setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(description='Standalone variant of object detection NetApp')
    parser.add_argument(
        '--detector',
        default="mmdetection",
        help="This argument is currently ignored."
        )
    args = parser.parse_args()
        
    logger.info("Starting Train Detector Service interface.")
    logger.info(f"The size of the queue set to: {NETAPP_INPUT_QUEUE}")

    # runs the flask server
    app.run(port=NETAPP_PORT, host='0.0.0.0')


if __name__ == '__main__':
    main()
