import os
from dataclasses import asdict

import pymongo

from celery_base import app
from data_models import RedditSubmission
from docker_logs import get_logger

logging = get_logger("worker-mongo")

mongo_uri = ''.join([
    'mongodb://',
    os.environ['MONGO_INITDB_ROOT_USERNAME'],
    ':',
    os.environ['MONGO_INITDB_ROOT_PASSWORD'],
    '@mongodb:27017'
])
# https://pymongo.readthedocs.io/en/stable/faq.html#is-pymongo-fork-safe
# myclient = pymongo.MongoClient(mongo_uri)
# mydb = myclient["reddit"]
# mycol = mydb["submissions"]


@app.task(bind=True, name='send_to_mongo')
def send_to_mongo(self, rSubmission: RedditSubmission):

    myclient = pymongo.MongoClient(mongo_uri)
    mydb = myclient["reddit"]
    mycol = mydb["submissions"]

    rSubmission.post_title_embedding = list(
        rSubmission.post_title_embedding.astype(float)
    )

    if len(rSubmission.post_text_embedding) > 0:
        rSubmission.post_text_embedding = list(
            rSubmission.post_text_embedding.astype(float)
        )

    try:
        _ = mycol.insert_one(asdict(rSubmission))
    except pymongo.errors.WriteError as e:
        logging.info('Cannot send item to MongoDB')
        logging.info(e)
    finally:
        logging.info('Item send to MongoDB')

    myclient.close()
