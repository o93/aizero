FROM python:3

MAINTAINER Hayato Okuda <oueiiti7@gmail.com>

RUN apt-get update && apt-get upgrade -y

RUN pip install numpy
RUN pip install matplotlib
RUN pip install Pillow

RUN apt-get clean
