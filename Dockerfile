FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

CMD ["bash", "runpod_job.sh"]
