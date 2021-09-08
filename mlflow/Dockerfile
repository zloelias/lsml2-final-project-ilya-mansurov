FROM pytorch/pytorch:latest
#FROM conda/miniconda3:latest

LABEL org.opencontainers.image.authors="zloelias@gmail.com"

#RUN apt-get update -y && \
#    apt-get install -y python3-pip python3-dev

# We copy just the requirements.txt first to leverage Docker cache
WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

COPY . /app

RUN chmod +x /app/start.sh

ENTRYPOINT [ "/app/start.sh" ]
