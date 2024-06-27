FROM python:3.11-buster as builder

RUN apt-get update && apt-get install -y git

RUN pip install -U pip

ENV HOST=0.0.0.0
ENV LISTEN_PORT 8080
EXPOSE 8080

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY ./app ./

CMD ["streamlit", "run", "app.py", "--server.port", "8080"]