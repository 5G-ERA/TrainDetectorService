
# 5G-Era Train Detector Service

This repository contains a service for detecting train movement in a video stream from a camera. It is intended to be used on autonomous or semi-autonomous robotic vehicle that operates inside an industrial area with a railway branch line (industrial spur). 

The robot can utilize this service each time it is about to cross a railway track on its route. Firstly, the robot must stop before the railway track, in order to allow the motion of the train to be distinguished from the movement of the video stream that is induced by the robot itself. Subsequently, Train Detector Service is called and it searches for any trains in the camera view and evaluates whether they are still or moving. This information can then be used by the robot to decide when it is safe to proceed with crossing the track.


### Docker 

Dockerfile is available in src/docker directory. Instructions for building and running docker image are in [docker readme](/src/docker/README.md).


### Local Installation

If you want to install the application locally, instead of using docker, see [local installation](local_installation.md) instructions.


### Client example

Client code example is provided in [client_no_middleware.py](/examples/python/client_no_middleware.py). To run the example script, following requirements needs to be installed: era-5g-client 0.4.1, simple-websocket

```bash
pip install era-5g-client==0.4.1 simple-websocket
```
### Related Repositories

- [era-5g-interface](https://github.com/5G-ERA/era-5g-interface) - Python interface (support classes) for Net Applications.
- [era-5g-client](https://github.com/5G-ERA/era-5g-client) - client classes for 5G-ERA Net Applications with various transport options.

