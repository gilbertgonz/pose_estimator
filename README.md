# Exploratory computer vision repo, nothin serious

Install [docker](https://docs.docker.com/engine/install/)

To build:
```
$ docker build -t camera_world .
```

To run:
```
$ xhost +local:docker
$ docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix camera_world
```