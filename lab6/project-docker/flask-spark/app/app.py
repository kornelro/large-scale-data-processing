# https://stackoverflow.com/questions/32719920/access-to-spark-from-flask-app

import numpy as np
from flask import Flask, request, abort

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf
from pyspark.ml import PipelineModel
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import StringIndexerModel

app = Flask(__name__)

sc = SparkContext('local')
sqlContext = SQLContext(sc)
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
lr_model = PipelineModel.load('/spark_models/lr_model')
bc_model = PipelineModel.load('/spark_models/bc_model')
mc_model = PipelineModel.load('/spark_models/mc_model')

# https://stackoverflow.com/questions/45885044/getting-labels-from-stringindexer-stages-within-pipeline-in-spark-pyspark
mc_classes = classes = {x._java_obj.getOutputCol(): x.labels 
                        for x in mc_model.stages if isinstance(x, StringIndexerModel)}
mc_classes = mc_classes['label']


@app.route('/lr')
def linear_regression():
    try:
        df = sqlContext.createDataFrame(
            [
                (
                    int(request.args.get('comments_number')), 
                    _str_to_list(request.args.get('post_title_embedding')), 
                    request.args.get('nsfw'), 
                    request.args.get('spoiler')
                )
            ],
            [
                'comments_number',
                'post_title_embedding',
                'nsfw',
                'spoiler'
            ]
        )
    except TypeError:
        abort(400, 'TypeError!')
    except ValueError:
        abort(400, 'ValueError!')

    if (
        request.args.get('nsfw') not in ['True', 'False']
    ) or (
        request.args.get('spoiler') not in ['True', 'False']
    ):
        abort(400, 'Wrong nsfw or spoiler value!')

    df = df.select(
        "comments_number",
        list_to_vector_udf(
            df["post_title_embedding"]
        ).alias("post_title_embedding"),
        'nsfw',
        'spoiler'
    )

    result = lr_model.transform(df)

    return str(
        max(
            int(result.collect()[0]['prediction']),
            0
        )
    )


@app.route('/bc')
def binary_classification():
    try:
        df = sqlContext.createDataFrame(
            [
                (
                    int(request.args.get('comments_number')), 
                    _str_to_list(request.args.get('post_title_embedding')), 
                    request.args.get('spoiler'),
                    int(request.args.get('up_votes_number'))
                )
            ],
            [
                'comments_number',
                'post_title_embedding',
                'spoiler',
                'up_votes_number'
            ]
        )
    except TypeError:
        abort(400, 'TypeError!')
    except ValueError:
        abort(400, 'ValueError!')

    if request.args.get('spoiler') not in ['True', 'False']:
        abort(400, 'Wrong spoiler value!')

    df = df.select(
        "comments_number",
        list_to_vector_udf(
            df["post_title_embedding"]
        ).alias("post_title_embedding"),
        'spoiler',
        'up_votes_number'
    )

    result = bc_model.transform(df)

    return str(int(result.collect()[0]['prediction']))


@app.route('/mc')
def multi_class_classification():
    try:
        df = sqlContext.createDataFrame(
            [
                (
                    int(request.args.get('comments_number')), 
                    _str_to_list(request.args.get('post_title_embedding')), 
                    request.args.get('spoiler'),
                    request.args.get('nsfw'), 
                    int(request.args.get('up_votes_number'))
                )
            ],
            [
                'comments_number',
                'post_title_embedding',
                'spoiler',
                'nsfw', 
                'up_votes_number'
            ]
        )
    except TypeError:
        abort(400, 'TypeError!')
    except ValueError:
        abort(400, 'ValueError!')

    if (
        request.args.get('nsfw') not in ['True', 'False']
    ) or (
        request.args.get('spoiler') not in ['True', 'False']
    ):
        abort(400, 'Wrong nsfw or spoiler value!')

    df = df.select(
        "comments_number",
        list_to_vector_udf(
            df["post_title_embedding"]
        ).alias("post_title_embedding"),
        'nsfw',
        'spoiler',
        'up_votes_number'
    )

    result = mc_model.transform(df)
    result = int(result.collect()[0]['prediction'])

    return mc_classes[result]


def _str_to_list(list_str):
    return list(
        map(
            lambda x: float(x),
            list_str[1:-1].split(', ')
        )
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
