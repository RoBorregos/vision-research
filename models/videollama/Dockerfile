# FROM nvidia/cuda:12.2.2-devel-ubuntu20.04
FROM ultralytics/yolov5:latest
RUN pip install --upgrade pip

# RUN apt-get update && apt-get install -y \
# 	python3-dev python3-pip git wget sudo

RUN pip install --upgrade pip

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN mkdir /videollama

WORKDIR /videollama
COPY . /videollama
RUN pip install -r requirements.txt
EXPOSE 8001
CMD ["python3", "main.py"]
