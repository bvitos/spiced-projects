FROM python:3.6-slim

WORKDIR /app

ADD requirements.txt /app

Add . /app

RUN pip install --trusted-host pypi.python.org -r requirements.txt

CMD ["python", "etl.py"]
#CMD ["python", "tweet_streamer2.py"]
