from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier


def get_linear_regression_pipeline():

    stringIndexer1 = StringIndexer(
        inputCol='nsfw',
        outputCol='nsfw_int'
    )

    stringIndexer2 = StringIndexer(
        inputCol='spoiler',
        outputCol='spoiler_int'
    )

    assembler = VectorAssembler(
        inputCols=[
            "post_title_embedding",
            "comments_number",
            "nsfw_int",
            "spoiler_int"
        ],
        outputCol="features"
    )

    lr = LinearRegression(
        labelCol='up_votes_number',
        featuresCol='features'
    )

    return Pipeline(
        stages=[
            stringIndexer1,
            stringIndexer2,
            assembler,
            lr
        ]
    )


def get_binary_classification_pipeline():

    stringIndexer1 = StringIndexer(
        inputCol='nsfw',
        outputCol='label'
    )

    stringIndexer2 = StringIndexer(
        inputCol='spoiler',
        outputCol='spoiler_int'
    )

    assembler = VectorAssembler(
        inputCols=[
            "post_title_embedding",
            "comments_number",
            "spoiler_int",
            "up_votes_number"
        ],
        outputCol="features"
    )

    lr = LogisticRegression(
        labelCol='label',
        featuresCol='features'
    )

    return Pipeline(
        stages=[
            stringIndexer1,
            stringIndexer2,
            assembler,
            lr
        ]
    )


def get_multi_classification_pipeline():

    stringIndexer1 = StringIndexer(
        inputCol='nsfw',
        outputCol='nsfw_int'
    )

    stringIndexer2 = StringIndexer(
        inputCol='spoiler',
        outputCol='spoiler_int'
    )

    stringIndexer3 = StringIndexer(
        inputCol='subbreddit_display_name',
        outputCol='label'
    )

    assembler = VectorAssembler(
        inputCols=[
            "post_title_embedding",
            "comments_number",
            "nsfw_int",
            "spoiler_int",
            "up_votes_number",
        ],
        outputCol="features"
    )

    dt = DecisionTreeClassifier(
        labelCol='label',
        featuresCol='features'
    )

    return Pipeline(
        stages=[
            stringIndexer1,
            stringIndexer2,
            stringIndexer3,
            assembler,
            dt
        ]
    )
