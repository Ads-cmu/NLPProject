# Ubuntu Linux as the base image
FROM ubuntu:22.04

# Set UTF-8 encoding
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install Python
RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get install -y git

# The following line ensures that subsequent install doesn't expect user input
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install python3-pip python3-dev

# Install spaCy
RUN pip3 install --upgrade pip
RUN pip3 install spacy
RUN python3 -m spacy download en_core_web_sm

# added later
RUN pip3 install transformers
RUN pip3 install sentencepiece
RUN pip3 install torch

RUN pip3 install protobuf==3.20.1

#For t5 QG model
RUN git clone https://huggingface.co/mrm8488/t5-base-finetuned-question-generation-ap

# Haystack for QA
RUN pip3 install farm-haystack

# # Install neural coref
RUN git clone https://github.com/huggingface/neuralcoref.git
RUN pip3 install -r /neuralcoref/requirements.txt
RUN pip3 install -e /neuralcoref/.

# RUN git clone https://github.com/huggingface/neuralcoref.git
# RUN cd ./neuralcoref
# RUN pip3 install -r requirements.txt
# RUN pip3 install -e .
# RUN cd ..

# Add the files into container, under QA folder, modify this based on your need
RUN mkdir /QA
ADD import_QG.py /QA
ADD import_QA.py /QA
ADD ask /QA
ADD answer /QA

RUN chmod 777 /QA/*
RUN chmod 777 /QA/answer
RUN chmod 777 /QA/ask
# Set working dir as /QA
WORKDIR /QA
RUN python3 import_QG.py
RUN python3 import_QA.py

ENTRYPOINT ["/bin/bash", "-c"]
