import os

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.ml.evaluation import RegressionEvaluator, \
    MulticlassClassificationEvaluator

from docker_logs import get_logger
from utils import get_class_distribution
from pipelines import get_linear_regression_pipeline, \
    get_binary_classification_pipeline, get_multi_classification_pipeline


logger = get_logger("app-spark")

conf = SparkConf().setAppName('app-spark').setMaster('local')
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

spark = SparkSession \
    .builder \
    .appName("app-spark") \
    .getOrCreate()

mongo_uri = ''.join([
    'mongodb://',
    os.environ['MONGO_INITDB_ROOT_USERNAME'],
    ':',
    os.environ['MONGO_INITDB_ROOT_PASSWORD'],
    '@mongodb:27017/reddit.submissions'
])

# data prepearation

df = spark.read.format("mongo").option("uri", mongo_uri).load()
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
bool_to_str_udf = udf(lambda l: str(l))
df_with_vectors = df.select(
    "_id",
    "up_votes_number",
    "comments_number",
    "subbreddit_display_name",
    "post_text",
    list_to_vector_udf(
        df["post_title_embedding"]
    ).alias("post_title_embedding"),
    bool_to_str_udf(
        df["nsfw"]
    ).alias("nsfw"),
    bool_to_str_udf(
        df["spoiler"]
    ).alias("spoiler")
)
train_data, test_data = df_with_vectors.randomSplit([0.75, 0.25], seed=1234)

logger.info('Data schema:')
logger.info(train_data.printSchema())
# logger.info(df.printSchema())

logger.info('Train data size:')
logger.info(train_data.count())
logger.info('Train data subreddits distribution:')
logger.info(get_class_distribution(train_data))

logger.info('Test data size:')
logger.info(test_data.count())
logger.info('Test data subreddits distribution:')
logger.info(get_class_distribution(test_data))

# regression
# up votes number prediction

lr_pipeline = get_linear_regression_pipeline()
lr_model = lr_pipeline.fit(train_data)
lr_pred_train = lr_model.transform(train_data)
lr_pred_test = lr_model.transform(test_data)
lr_evaluator = RegressionEvaluator(
    predictionCol="prediction",
    labelCol="up_votes_number",
    metricName="rmse"
)

logger.info(
    'Regression train RMSE: %g' %
    lr_evaluator.evaluate(lr_pred_train)
)
logger.info(
    'Regression test RMSE: %g' %
    lr_evaluator.evaluate(lr_pred_test)
)

# save pipeline

lr_model.save('/app/saved_models/lr_model')

# binary classification
# predict if post belongs to AskReddit

bc_pipeline = get_binary_classification_pipeline()
bc_model = bc_pipeline.fit(train_data)
bc_pred = bc_model.transform(test_data)
bc_evaluator_acc = MulticlassClassificationEvaluator(
    predictionCol="prediction",
    labelCol="label",
    metricName="accuracy"
)
bc_evaluator_f1 = MulticlassClassificationEvaluator(
    predictionCol="prediction",
    labelCol="label",
    metricName="f1"
)

logger.info(
    'Binary classification test Accuracy: %g' %
    bc_evaluator_acc.evaluate(bc_pred)
)
logger.info(
    'Binary classification test F1: %g' %
    bc_evaluator_f1.evaluate(bc_pred)
)

# save pipeline

bc_model.save('/app/saved_models/bc_model')

# multi-class classification
# predict post's subreddit

mc_pipeline = get_multi_classification_pipeline()
mc_model = mc_pipeline.fit(train_data)
mc_pred = mc_model.transform(test_data)
mc_evaluator_acc = MulticlassClassificationEvaluator(
    predictionCol="prediction",
    labelCol="label",
    metricName="accuracy"
)
mc_evaluator_f1 = MulticlassClassificationEvaluator(
    predictionCol="prediction",
    labelCol="label",
    metricName="f1"
)

logger.info(
    'Multi-class classification test Accuracy: %g' %
    mc_evaluator_acc.evaluate(mc_pred)
)
logger.info(
    'Multi-class classification test F1: %g' %
    mc_evaluator_f1.evaluate(mc_pred)
)

# save pipeline

mc_model.save('/app/saved_models/mc_model')


# some additional sources
# https://towardsdatascience.com/building-a-linear-regression-with-pyspark-and-mllib-d065c3ba246a
# https://towardsdatascience.com/machine-learning-with-pyspark-and-mllib-solving-a-binary-classification-problem-96396065d2aa
