FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /workspace/

# install basics
RUN apt-get update -y
RUN apt-get install -y git curl ca-certificates bzip2 cmake tree htop bmon iotop sox libsox-dev libsox-fmt-all vim

# install python deps
RUN pip install cython visdom cffi tensorboardX wget jupyter

ENV CUDA_HOME=/usr/local/cuda

# install ctcdecode
RUN git clone --recursive https://github.com/parlance/ctcdecode.git
RUN cd ctcdecode; pip install .

# install apex
RUN git clone --recursive https://github.com/NVIDIA/apex.git
RUN cd apex; pip install .

COPY requirements.txt .
RUN pip install -r requirements.txt

ADD data data
ADD deepspeech_pytorch deepspeech_pytorch
COPY .comet.config labels.json setup.py train.py ./
RUN pip install -e .

RUN jupyter serverextension enable --py jupyter_http_over_ws

# launch jupyter
CMD jupyter-lab --ip 0.0.0.0 --no-browser --allow-root --LabApp.token=''
