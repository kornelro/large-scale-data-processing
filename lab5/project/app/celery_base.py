import os
import praw
from pymagnitude import Magnitude
from celery import Celery
from influxdb import InfluxDBClient

from docker_logs import get_logger


logging = get_logger("celery-base")

# prepare env CELERY_BROKER_URL
os.environ['CELERY_BROKER_URL'] = ''.join([
    os.environ['broker_protocol'],
    '://',
    os.environ['broker_user'],
    ':',
    os.environ['broker_password'],
    '@',
    os.environ['broker_host'],
    ':',
    os.environ['broker_port']
])

# prepare celery app
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
influxdb_host = os.environ['influxdb_host']
influxdb_user = os.environ['influxdb_user']
influxdb_password = os.environ['influxdb_password']
influxdb_client = InfluxDBClient(
    influxdb_host,
    8086,
    influxdb_user,
    influxdb_password,
    'celery'
)
influxdb_client.create_database('celery')
logging.info('App connected with influxdb')
