import base64
import binascii
import secrets
from queue import Queue
import cv2
import argparse
import os
import numpy as np
import flask_socketio
import logging
import time

from dataclasses import dataclass
from flask import Flask, Response, request, session
from flask_session import Session
from typing import Dict

from era_5g_train_detection_standalone.worker import TrainDetectorWorker
from era_5g_interface.task_handler import TaskHandler
from era_5g_interface.task_handler_gstreamer_internal_q import \
    TaskHandlerGstreamerInternalQ, TaskHandlerGstreamer
from era_5g_interface.task_handler_internal_q import TaskHandlerInternalQ

# port of the netapp's server
NETAPP_PORT = os.getenv("NETAPP_PORT", 5896)

# flask initialization
app = Flask(__name__)
app.secret_key = secrets.token_hex()
app.config['SESSION_TYPE'] = 'filesystem'

# socketio initialization for sending results to the client
Session(app)
socketio = flask_socketio.SocketIO(app, manage_session=False, async_mode='threading')

# list of available ports for gstreamer communication - they need to be exposed when running in docker / k8s
# could be changed using the script arguments
free_ports = [5001, 5002, 5003]

@dataclass
class TaskAndWorker:
    task: TaskHandler
    worker: TrainDetectorWorker

# list of registered tasks
tasks: Dict[str, TaskAndWorker] = dict()


class ArgFormatError(Exception):
    pass


@app.route('/register', methods=['POST'])
def register():
    """
    Needs to be called before an attempt to open WS is made.

    Returns:
        _type_: The port used for gstreamer communication.
    """
    # print(f"register {session.sid} {session}")
    args = request.get_json(silent=True)
    gstreamer = False
    if args:
        gstreamer = args.get("gstreamer", False)

    if gstreamer and not free_ports:
        return {"error": "Not enough resources"}, 503

    session['registered'] = True

    # new queue for received images
    image_queue = Queue(30)

    # select the appropriate task handler, depends on whether the client wants to use
    # gstreamer to pass the images or not 
    if gstreamer:
        port = free_ports.pop(0)
        task = TaskHandlerGstreamerInternalQ(session.sid, port, image_queue, daemon=True)
    else:
        task = TaskHandlerInternalQ(session.sid, image_queue, daemon=True)

    # create worker and run it as thread, listening to image_queue
    worker = TrainDetectorWorker(image_queue, app, daemon=True)
    worker.start()

    tasks[session.sid] = TaskAndWorker(task, worker)

    print(f"Client registered: {session.sid}")
    if gstreamer:
        return {"port": port}, 200
    else:
        return Response(status=204)


@app.route('/unregister', methods=['POST'])
def unregister():
    """_
    Disconnects the websocket and removes the task from the memory.

    Returns:
        _type_: 204 status
    """
    session_id = session.sid
    # print(f"unregister {session.sid} {session}")
    # print(f"{request}")
    if session.pop('registered', None):
        task_and_worker = tasks.pop(session.sid)
        task = task_and_worker.task
        worker = task_and_worker.worker
        task.stop()
        worker.stop()
        flask_socketio.disconnect(task.websocket_id, namespace="/results")
        if isinstance(task, TaskHandlerGstreamer):
            free_ports.append(task.port)
        print(f"Client unregistered: {session_id}")

    return Response(status=204)


@app.route('/image', methods=['POST'])
def image_callback_http():
    """
    Allows to receive jpg-encoded image using the HTTP transport
    """

    recv_timestamp = time.time_ns()

    if 'registered' not in session:
        return Response('Need to call /register first.', 401)

    sid = session.sid
    task = tasks[sid].task

    if "timestamps[]" in request.args:
        timestamps = request.args.to_dict(flat=False)['timestamps[]']
    else:
        timestamps = []
    # convert string of image data to uint8
    index = 0
    for file in request.files.to_dict(flat=False)['files']:
        # print(file)
        nparr = np.frombuffer(file.read(), np.uint8)

        # store the image to the appropriate task
        # the image is not decoded here to make the callback as fast as possible
        task.store_image(
            {"sid": sid,
             "websocket_id": task.websocket_id,
             "timestamp": timestamps[index],
             "recv_timestamp": recv_timestamp,
             "decoded": False},
            nparr
            )
        index += 1
    return Response(status=204)


@socketio.on('connect', namespace='/data')
def connect_data(auth):
    """_summary_
    Creates a websocket connection to the client for passing the data.

    Raises:
        ConnectionRefusedError: Raised when attempt for connection were made
            without registering first.
    """

    if 'registered' not in session:
        raise ConnectionRefusedError('Need to call /register first.')

    print(f"Connected data. Session id: {session.sid}, ws_sid: {request.sid}")

    flask_socketio.send("you are connected", namespace='/data', to=request.sid)


@socketio.on('image', namespace='/data')
def image_callback_websocket(data: dict):
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
    recv_timestamp = time.time_ns()
    
    if 'timestamp' in data:
        timestamp = data['timestamp']
    else:
        logging.debug("Timestamp not set, setting default value")
        timestamp = 0
    if 'registered' not in session:
        logging.error(f"Non-registered client tried to send data")
        flask_socketio.emit(
            "image_error",
            {"timestamp": timestamp,
             "error": "Need to call /register first."},
            namespace='/data',
            to=request.sid
            )
        return
    if 'frame' not in data:
        logging.error(f"Data does not contain frame.")
        flask_socketio.emit(
            "image_error",
            {"timestamp": timestamp,
             "error": "Data does not contain frame."},
            namespace='/data',
            to=request.sid
            )
        return

    sid = session.sid
    task = tasks[sid].task

    try:
        frame = base64.b64decode(data["frame"])
        task.store_image(
            {"sid": session.sid,
             "websocket_id": task.websocket_id,
             "timestamp": timestamp,
             "recv_timestamp": recv_timestamp,
             "decoded": False},
            np.frombuffer(frame, dtype=np.uint8)
            )
    except (ValueError, binascii.Error) as error:
        logging.error(f"Failed to decode frame data: {error}")
        flask_socketio.emit(
            "image_error",
            {"timestamp": timestamp,
             "error": f"Failed to decode frame data: {error}"},
            namespace='/data',
            to=request.sid
            )


@socketio.on('json', namespace='/data')
def json_callback_websocket(data):
    """
    Allows to receive general json data using the websocket transport

    Args:
        data (dict): NetApp-specific json data

    Raises:
        ConnectionRefusedError: Raised when attempt for connection were made
            without registering first.
    """
    if 'registered' not in session:
        logging.error(f"Non-registered client tried to send data")
        flask_socketio.emit(
            "json_error",
            {"error": "Need to call /register first."},
            namespace='/data',
            to=request.sid
            )
    logging.debug(f"client with task id: {session.sid} sent data {data}")


@socketio.on('connect', namespace='/results')
def connect_results(auth):
    """
    Creates a websocket connection to the client for passing the results.

    Raises:
        ConnectionRefusedError: Raised when attempt for connection were made
            without registering first.
    """

    # print(f"connect {session.sid} {session}")
    # print(f"{request.sid} {request}")
    if 'registered' not in session:
        # TODO: disconnect?
        # flask_socketio.disconnect(request.sid, namespace="/results")
        raise ConnectionRefusedError('Need to call /register first.')

    sid = request.sid
    print(f"Client connected: session id: {session.sid}, websocket id: {sid}")
    tasks[session.sid].task.websocket_id = sid
    if isinstance(tasks[session.sid].task, TaskHandlerGstreamer):
        tasks[session.sid].task.start()
    # TODO: Check task is running, Gstreamer capture can failed
    flask_socketio.send("You are connected", namespace='/results', to=sid)


@socketio.on('disconnect', namespace='/results')
def disconnect_results():
    print(f"Client disconnected from /results namespace: session id: {session.sid}, websocket id: {request.sid}")


@socketio.on('disconnect', namespace='/data')
def disconnect_data():
    print(f"Client disconnected from /data namespace: session id: {session.sid}, websocket id: {request.sid}")


def get_ports_range(ports_range) -> list:
    """
    Decodes port range in format port_1:port_n

    Args:
        ports_range (str): range of ports

    Raises:
        ArgFormatError: raised when ports range is in incorrect format

    Returns:
        list: list of ports
    """
    if ports_range.count(':') != 1:
        raise ArgFormatError
    r1, r2 = ports_range.split(':')
    if int(r2) <= int(r1):
        raise ArgFormatError
    return [port for port in range(int(r1), int(r2) + 1)]


def main(args=None):
    logging.getLogger().setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(description='Standalone variant of object detection NetApp')
    # TODO: in future versions, the ports should be specified as list instead of range, preferably
    #       using env variable, for compatibility with middleware
    parser.add_argument(
        '--ports',
        default="5001:5003",
        help="Specify the range of ports available for gstreamer connections. Format "
             "port_start:port_end. Default is 5001:5003."
        )
    parser.add_argument(
        '--detector',
        default="fps",
        help="Select detector. Available options are opencv, mmdetection, fps. Default is fps."
        )
    args = parser.parse_args()
    global free_ports
    try:
        free_ports = get_ports_range(args.ports)
    except ArgFormatError:
        logging.error("Port range specified in wrong format. The correct format is port_start:port_end, e.g. 5001:5003.")
        exit()
    
    logging.info("Starting Train Detector Service interface.")

    # runs the flask server
    # allow_unsafe_werkzeug needs to be true to run inside the docker
    # TODO: use better webserver
    socketio.run(app, port=NETAPP_PORT, host='0.0.0.0', allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
