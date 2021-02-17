import os

import streamlit as st

from api import call_bc, call_embedder, call_lr, call_mc


FLASK_EMBEDDER_SERVICE_HOST = os.environ['FLASK_EMBEDDER_SERVICE_HOST']
FLASK_EMBEDDER_SERVICE_PORT = os.environ['FLASK_EMBEDDER_SERVICE_PORT']
FLASK_SPARK_SERVICE_HOST = os.environ['FLASK_SPARK_SERVICE_HOST']
FLASK_SPARK_SERVICE_PORT = os.environ['FLASK_SPARK_SERVICE_PORT']
LR_ROUTING = os.environ['LR_ROUTING']
BC_ROUTING = os.environ['BC_ROUTING']
MC_ROUTING = os.environ['MC_ROUTING']


def render_lr():
    st.markdown('Predict number of post up votes.')

    post_title = st.text_input(
        label='Post title'
    )
    comments_number = st.number_input(
        label='Comments number',
        min_value=0
    )
    nsfw = st.checkbox(
        label='NSFW'
    )
    spoiler = st.checkbox(
        label='Spoiler'
    )

    if st.button('Run'):
        if len(post_title) > 0:

            post_title_embedding = call_embedder(
                text=post_title,
                host=FLASK_EMBEDDER_SERVICE_HOST,
                port=FLASK_EMBEDDER_SERVICE_PORT
            )

            prediction = call_lr(
                comments_number=comments_number,
                post_title_embedding=post_title_embedding,
                nsfw=nsfw,
                spoiler=spoiler,
                host=FLASK_SPARK_SERVICE_HOST,
                port=FLASK_SPARK_SERVICE_PORT,
                routing=LR_ROUTING
            )
            st.markdown('Predicted up votes number: '+prediction)

        else:
            st.markdown('Post title cannot be empty!')

def render_bc():
    st.markdown('Predict if post is NSFW.')

    post_title = st.text_input(
        label='Post title'
    )
    comments_number = st.number_input(
        label='Comments number',
        min_value=0
    )
    up_votes_number = st.number_input(
        label='Up votes number',
        min_value=0
    )
    spoiler = st.checkbox(
        label='Spoiler'
    )

    if st.button('Run'):
        if len(post_title) > 0:

            post_title_embedding = call_embedder(
                text=post_title,
                host=FLASK_EMBEDDER_SERVICE_HOST,
                port=FLASK_EMBEDDER_SERVICE_PORT
            )

            prediction = call_bc(
                comments_number=comments_number,
                post_title_embedding=post_title_embedding,
                spoiler=spoiler,
                up_votes_number=up_votes_number,
                host=FLASK_SPARK_SERVICE_HOST,
                port=FLASK_SPARK_SERVICE_PORT,
                routing=BC_ROUTING
            )
            st.markdown('NSFW prediction: '+prediction)

        else:
            st.markdown('Post title cannot be empty!')

def render_mc():
    st.markdown('Predict post SubReddit.')

    post_title = st.text_input(
        label='Post title'
    )
    comments_number = st.number_input(
        label='Comments number',
        min_value=0
    )
    up_votes_number = st.number_input(
        label='Up votes number',
        min_value=0
    )
    nsfw= st.checkbox(
        label='NSFW'
    )
    spoiler = st.checkbox(
        label='Spoiler'
    )

    if st.button('Run'):
        if len(post_title) > 0:

            post_title_embedding = call_embedder(
                text=post_title,
                host=FLASK_EMBEDDER_SERVICE_HOST,
                port=FLASK_EMBEDDER_SERVICE_PORT
            )

            prediction = call_mc(
                comments_number=comments_number,
                post_title_embedding=post_title_embedding,
                nsfw=nsfw,
                spoiler=spoiler,
                up_votes_number=up_votes_number,
                host=FLASK_SPARK_SERVICE_HOST,
                port=FLASK_SPARK_SERVICE_PORT,
                routing=MC_ROUTING
            )
            st.markdown('Predicted SubReddit: '+prediction)

        else:
            st.markdown('Post title cannot be empty!')
