FROM pytorch/pytorch

ENV DEBIAN_FRONTEND=noninteractive

COPY ./entrypoint.sh /entrypoint.sh

RUN apt-get update -y
RUN apt-get update && apt-get install vim -y
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get clean -y

RUN pip install albumentations==1.0.3

RUN apt-get update && \
    apt-get -y install gcc mono-mcs && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean -y

RUN pip install effdet==0.3.0
RUN pip install matplotlib==3.5.2
RUN pip install numpy
RUN pip install opencv_python==4.6.0.66
RUN pip install pytorch_lightning==1.8.6
RUN pip install scipy==1.5.4
RUN pip install torch==1.12.1
RUN pip install torchvision==0.13.1
RUN pip install tqdm==4.63.0

RUN chmod 0777 /entrypoint.sh
#CMD ["/entrypoint.sh"]
RUN mkdir /home/metric-user
RUN chmod -R 0777 /home/metric-user
RUN /entrypoint.sh
