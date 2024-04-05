# Exploratory computer vision repo, nothin serious

Install [docker](https://docs.docker.com/engine/install/)

## pose_estimator/

To build:
```
$ docker build -t pose_estimator .
```

To run:
```
$ xhost +local:docker
$ docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix pose_estimator
```

## sports_ml_tracking/

To build:
```
$ docker build -t sports_ml_tracking .
```

To run:
```
$ xhost +local:docker
$ docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix sports_ml_tracking
```