FROM nvcr.io/nvidia/pytorch:23.10-py3

ENV DEBIAN_FRONTEND=noninteractive

# Update the package list
RUN apt-get update -y

# Install git
RUN apt-get install -y git

RUN python -m pip install --upgrade pip
ADD requirements.txt .
#RUN pip install -r requirements.txt