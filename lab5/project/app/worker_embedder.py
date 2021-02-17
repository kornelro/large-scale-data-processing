from pymagnitude import Magnitude
import numpy as np

from celery_base import app
from data_models import RedditSubmission
from docker_logs import get_logger


logger = get_logger("worker-embedder")


@app.task(bind=True, name='put_embeddings')
def put_embeddings(self, rSubmission: RedditSubmission):
    vecs = Magnitude('word2vec/light/GoogleNews-vectors-negative300')
    # vecs = Magnitude('http://magnitude.plasticity.ai/word2vec/light/GoogleNews-vectors-negative300.magnitude')

    rSubmission.post_title_embedding = np.mean(
        vecs.query(
            rSubmission.post_title.split()
        ),
        axis=0
    )

    if len(rSubmission.post_text) > 0:
        rSubmission.post_text_embedding = np.mean(
            vecs.query(
                rSubmission.post_text.split()
            ),
            axis=0
        )

    logger.info('Embedded submission: ', rSubmission.post_title)

    return rSubmission
