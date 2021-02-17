import numpy as np
from flask import request
from pymagnitude import Magnitude

from app import app

vecs = Magnitude('word2vec/light/GoogleNews-vectors-negative300')

@app.route('/')
def home():
    text = request.args.get('text')

    if text:
        embedding = np.mean(
            vecs.query(
                text.split()
            ),
            axis=0
        )
    else:
        embedding = ''

    return str(embedding)
