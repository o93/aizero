FROM python:3

MAINTAINER Hayato Okuda <okuda@a-tm.co.jp>

RUN apt-get update && apt-get upgrade -y

RUN pip install numpy
RUN pip install matplotlib
RUN pip install Pillow

RUN apt-get clean
