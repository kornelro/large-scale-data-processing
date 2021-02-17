import pyspark.sql.functions as F
from pyspark.sql.types import FloatType


def get_class_distribution(
    df,
    col_name='subbreddit_display_name'
):
    count = df.groupBy(col_name).agg(F.countDistinct('_id'))
    s = count.select(F.sum('count(DISTINCT _id)')).collect()[0][0]

    udf_func = F.udf(
        lambda col: round(col / s, 4) * 100,
        FloatType()
    )

    count = count.withColumn(
        'set_part_%',
        udf_func('count(DISTINCT _id)')
    )

    return count.show()
