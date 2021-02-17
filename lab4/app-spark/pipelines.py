from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier

from tranformers import AddBinaryLabelColumn, AddHasText


def get_linear_regression_pipeline():

    assembler = VectorAssembler(
        inputCols=[
            "post_title_embedding",
            "comments_number",
            "nsfw",
            "spoiler"
        ],
        outputCol="features"
    )

    lr = LinearRegression(
        labelCol='up_votes_number',
        featuresCol='features'
    )

    return Pipeline(stages=[assembler, lr])


def get_binary_classification_pipeline():

    transformer = AddBinaryLabelColumn()

    assembler = VectorAssembler(
        inputCols=[
            "post_title_embedding",
            "comments_number",
            "nsfw",
            "spoiler",
            'up_votes_number'
        ],
        outputCol="features"
    )

    lr = LogisticRegression(
        labelCol='label',
        featuresCol='features'
    )

    return Pipeline(
        stages=[
            transformer,
            assembler,
            lr
        ]
    )


def get_multi_classification_pipeline():

    transformer = AddHasText()

    stringIndexer = StringIndexer(
        inputCol='subbreddit_display_name',
        outputCol='label'
    )

    assembler = VectorAssembler(
        inputCols=[
            "post_title_embedding",
            "comments_number",
            "nsfw",
            "spoiler",
            "up_votes_number",
            "has_text"
        ],
        outputCol="features"
    )

    dt = DecisionTreeClassifier(
        labelCol='label',
        featuresCol='features'
    )

    return Pipeline(
        stages=[
            transformer,
            stringIndexer,
            assembler,
            dt
        ]
    )
