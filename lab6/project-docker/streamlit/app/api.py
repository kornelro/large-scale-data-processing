from typing import List
import re

import numpy as np
import requests

import logging


def call_embedder(
    text: str,
    host: str = 'flask-embedder',
    port: str = '80'
):
    url = 'http://' + host + ':' + port + '/'

    text = re.sub(r'[^\w\s]', '', text)

    response = requests.get(
        url=url,
        params={'text':text},
    )

    embedding = re.sub('\s+', ' ', response.text)
    embedding = embedding[1:-1].split(' ')
    while '' in embedding:
        embedding.remove('')
    embedding = list(
        map(
            lambda x: float(x),
            embedding
        )
    )

    return embedding


def call_lr(
    comments_number: int,
    post_title_embedding: List[float],
    nsfw: bool,
    spoiler: bool,
    host: str = 'flask-spark',
    port: str = '8080',
    routing: str = 'lr'
):
    url = 'http://' + host + ':' + port + '/' + routing

    response = requests.get(
        url=url,
        params={
            'comments_number': comments_number,
            'post_title_embedding': str(post_title_embedding),
            'nsfw': str(nsfw),
            'spoiler': str(spoiler)
        },
    )

    return response.text


def call_bc(
    comments_number: int,
    post_title_embedding: List[float],
    spoiler: bool,
    up_votes_number: int,
    host: str = 'flask-spark',
    port: str = '8080',
    routing: str = 'bc'
):
    url = 'http://' + host + ':' + port + '/' + routing

    response = requests.get(
        url=url,
        params={
            'comments_number': comments_number,
            'post_title_embedding': str(post_title_embedding),
            'spoiler': str(spoiler),
            'up_votes_number': up_votes_number
        },
    )

    if response.text == "1":
        result = "Yes"
    else:
        result = "No"

    return result


def call_mc(
    comments_number: int,
    post_title_embedding: List[float],
    nsfw: bool,
    spoiler: bool,
    up_votes_number: int,
    host: str = 'flask-spark',
    port: str = '8080',
    routing: str = 'mc'
):
    url = 'http://' + host + ':' + port + '/' + routing

    response = requests.get(
        url=url,
        params={
            'comments_number': comments_number,
            'post_title_embedding': str(post_title_embedding),
            'nsfw': str(nsfw),
            'spoiler': str(spoiler),
            'up_votes_number': up_votes_number
        },
    )

    return response.text
