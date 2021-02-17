from pyspark.ml import Transformer
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType


class AddBinaryLabelColumn(Transformer):

    def _transform(self, dataset):

        udf_func = F.udf(
            lambda col: 1 if col == 'AskReddit' else 0,
            IntegerType()
        )

        df_with_label = dataset.withColumn(
            'label',
            udf_func('subbreddit_display_name')
        )

        return df_with_label


class AddHasText(Transformer):

    def _transform(self, dataset):

        udf_func = F.udf(
            lambda col: 1 if len(col) > 0 else 0,
            IntegerType()
        )

        df = dataset.withColumn(
            'has_text',
            udf_func('post_text')
        )

        return df
