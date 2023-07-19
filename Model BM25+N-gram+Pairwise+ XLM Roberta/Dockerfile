FROM python:3.7

WORKDIR /code
COPY requirements.txt /code

RUN apt update; apt-get -y install libgl1-mesa-glx libglib2.0-0 vim
RUN apt-get -y install python3-dev
RUN apt-get -y install default-jdk default-jre
RUN pip install --upgrade pip
RUN pip install Cython
RUN pip install jupyterlab
RUN pip install numpy
RUN pip install -r requirements.txt

COPY . /code

WORKDIR /
