# FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

FROM python:3.9

# Set environment variables
# ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        # python3.9 \
        # python3-pip \
        python3-opencv \
        # python3.9-venv \
        libglib2.0-0

# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN mkdir /app
WORKDIR /app

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


# RUN echo $(python3 --version)
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN pip3 install numpy \
                gymnasium \
                wheel \
                kaggle-environments

# COPY ./requirements.txt .
COPY pyproject.toml ./pyproject.toml
COPY connect_x ./connect_x

RUN pip3 install -e .

CMD ["python", "connect_x/main.py"]


