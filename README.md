# Exploring camera pose estimation applications

Install [docker](https://docs.docker.com/engine/install/)

To build:
```
$ docker build -t pose_estimator .
```

To run:
```
$ xhost +local:docker
$ docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix pose_estimator
```