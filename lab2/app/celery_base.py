import os
import time
from datetime import datetime, timedelta

import influxdb
import praw
import prawcore
from celery import Celery
from influxdb import InfluxDBClient

from docker_logs import get_logger


logging = get_logger("task")
app = Celery()
influxdb_client = InfluxDBClient('influxdb', 8086, 'root', 'root', 'celery')
influxdb_client.create_database('celery')
reddit = praw.Reddit(
    user_agent="studentproject by u/"+os.environ['praw_username']
)
logging.info('App connected with Reddit user: '+str(reddit.user.me()))


# The add_periodic_task() function will add the entry
# to the beat_schedule setting behind the scenes
@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    sender.add_periodic_task(
        int(os.environ['frequency_s']),
        get_subreddit.s(),
        name='subreddits'
    )


@app.task(bind=True, name='get_subreddit')
def get_subreddit(self):
    subreddit_name = os.environ['subreddit']
    time_diff = int(os.environ['frequency_s'])
    current_time = datetime.utcnow()
    time_lower_bound = current_time - timedelta(seconds=time_diff)

    new_submissions = []
    json_metrics = []
    try:
        submissions = reddit.subreddit(subreddit_name).new()
        new_submissions = [
            sub.id for sub in submissions
            if datetime.utcfromtimestamp(sub.created_utc) >= time_lower_bound
        ]
    except prawcore.exceptions.PrawcoreException:
        logging.info('Cannot read subreddit from Reddit API.')
    finally:
        json_metrics = [
            {
                "measurement": "new_submissions",
                "time": str(datetime.utcnow()),
                "fields": {
                    "new_submissions": len(new_submissions)
                }
            }
        ]

    try:
        influxdb_client.write_points(json_metrics)
    except influxdb.exceptions.InfluxDBServerError:
        logging.info('Cannot send metrics to InfluxDB - Server Error.')
    except influxdb.exceptions.InfluxDBClientError:
        logging.info('Cannot send metrics to InfluxDB - Client Error.')

    if len(new_submissions) == 0:
        logging.info('No new submissions.')
    else:
        for submission_id in new_submissions:
            get_submission.delay(submission_id)


@app.task(bind=True, name='get_submission')
def get_submission(self, submission_id):

    json_metrics = []
    try:
        start_time = time.time()
        submission = reddit.submission(id=submission_id)
        fetch_time = time.time() - start_time
    except prawcore.exceptions.PrawcoreException:
        logging.info('Cannot get submission from Reddit API.')
    finally:
        title = submission.title
        title = (title[:15] + '..') if len(title) > 15 else title
        logging.info('SUBMISSION TITLE: '+title)
        json_metrics = [
            {
                "measurement": "submissions",
                "time": str(datetime.now()),
                "fields": {
                    "fetch_time": fetch_time,
                    "title_len": len(submission.title),
                    "text_len": len(submission.selftext),
                    "nsfw": str(submission.over_18),
                    "spoiler": str(submission.spoiler),
                    "original_content": str(submission.is_original_content),

                }
            }
        ]

    try:
        influxdb_client.write_points(json_metrics)
    except influxdb.exceptions.InfluxDBServerError:
        logging.info('Cannot send metrics to InfluxDB - Server Error.')
    except influxdb.exceptions.InfluxDBClientError:
        logging.info('Cannot send metrics to InfluxDB - Client Error.')
