import time
from datetime import datetime, timedelta
import os

import influxdb
import prawcore
from celery import signature

from celery_base import influxdb_client, reddit, app
from docker_logs import get_logger
from data_models import RedditSubmission


logging = get_logger("worker-scraper")


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
        submission = RedditSubmission(
            original_URL=submission.url,
            author_name=submission.author.name,
            subreddit_name=submission.subreddit.name,
            subbreddit_display_name=submission.subreddit.display_name,
            post_title=submission.title,
            post_title_embedding=[],
            post_text=submission.selftext,
            post_text_embedding=[],
            upvote_ratio=submission.upvote_ratio,
            up_votes_number=submission.score,
            comments_number=submission.num_comments,
            nsfw=submission.over_18,
            spoiler=submission.spoiler,
            original=submission.is_original_content,
            distinguished=submission.distinguished,
            locked=submission.locked,
            fetch_time=fetch_time
        )

        title = submission.post_title
        title_to_log = (title[:15] + '..') if len(title) > 15 else title
        logging.info('SUBMISSION TITLE: ' + title_to_log)

        x = signature(
            'put_embeddings', args=[submission]
        ) | signature('send_to_mongo')
        x.apply_async()

        json_metrics = [
            {
                "measurement": "submissions",
                "time": str(datetime.now()),
                "fields": {
                    "fetch_time": fetch_time,
                    "title_len": len(submission.post_title),
                    "text_len": len(submission.post_text),
                    "nsfw": str(submission.nsfw),
                    "spoiler": str(submission.spoiler),
                    "original_content": str(submission.original),

                }
            }
        ]

    try:
        influxdb_client.write_points(json_metrics)
    except influxdb.exceptions.InfluxDBServerError:
        logging.info('Cannot send metrics to InfluxDB - Server Error.')
    except influxdb.exceptions.InfluxDBClientError:
        logging.info('Cannot send metrics to InfluxDB - Client Error.')
