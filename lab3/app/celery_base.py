import os
import praw
from pymagnitude import Magnitude
from celery import Celery
from influxdb import InfluxDBClient

from docker_logs import get_logger


logging = get_logger("celery-base")

app = Celery()
app.conf.update({
    'task_routes': {
        'get_subreddit': {'queue': 'scraper'},
        'get_submission': {'queue': 'scraper'},
        'put_embeddings': {'queue': 'embedder'},
        'send_to_mongo': {'queue': 'mongo'}
    },
    'task_serializer': 'pickle',
    'result_serializer': 'pickle',
    'accept_content': ['pickle']
})

# connect to reddit
reddit = praw.Reddit(
    user_agent="studentproject by u/" + os.environ['praw_username']
)
logging.info('App connected with Reddit user: ' + str(reddit.user.me()))

# connect to influx
influxdb_client = InfluxDBClient('influxdb', 8086, 'root', 'root', 'celery')
influxdb_client.create_database('celery')
logging.info('App connected with influxdb')
