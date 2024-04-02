FROM ubuntu:jammy as build

COPY requirements.txt /requirements.txt

RUN apt update && apt install -y \ 
    python3-pip libgl1 libglib2.0-0 x11-apps \
    && pip install -r requirements.txt

COPY calibration/ /app/calibration
COPY test_assets/ /app/test_assets
COPY pose_estimator.py /app/pose_estimator.py

### Final stage build
FROM scratch

COPY --from=build / /

WORKDIR /app

ENTRYPOINT ["./pose_estimator.py"]