"""
spark_streaming_example.py
---------------------------
OPTIONAL advanced module. Example of streaming transaction files through
Spark Structured Streaming and scoring them with a pre-trained Spark ML
model. NOT required for the core project.

Install: pip install pyspark

Fixes vs. original spec:
- getOrCreate() was being called without parentheses -> `.getOrCreate()`
- "stimstamp" typo corrected to "timestamp"
- StringIndexer stage added before VectorAssembler (indexed_df was
  referenced but never created in the original spec)
- Missing imports (StringIndexer, functions.hour) added
"""
from __future__ import annotations

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from pyspark.sql.functions import hour
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel


def run_spark_streaming_example(input_dir: str = "streaming_input/", model_path: str = "models/fraud_model_spark"):
    spark = (
        SparkSession.builder
        .appName("FraudDetectionSystem")
        .getOrCreate()
    )

    schema = (
        StructType()
        .add("transaction_id", "integer")
        .add("customer_id", "string")
        .add("amount", "double")
        .add("timestamp", "timestamp")
        .add("location", "string")
        .add("device", "string")
        .add("is_fraud", "integer")
    )

    transactions = (
        spark.readStream
        .schema(schema)
        .option("maxFilesPerTrigger", 1)
        .csv(input_dir)
    )

    # Feature engineering
    transactions = transactions.withColumn("hour", hour("timestamp"))

    # Encode categorical features (fixes: indexed_df was undefined in original spec)
    location_indexer = StringIndexer(inputCol="location", outputCol="loc_idx", handleInvalid="keep")
    device_indexer = StringIndexer(inputCol="device", outputCol="dev_idx", handleInvalid="keep")

    indexed_df = location_indexer.fit(transactions).transform(transactions)
    indexed_df = device_indexer.fit(indexed_df).transform(indexed_df)

    assembler = VectorAssembler(
        inputCols=["amount", "hour", "loc_idx", "dev_idx"],
        outputCol="features",
    )
    feature_df = assembler.transform(indexed_df)

    model = LogisticRegressionModel.load(model_path)
    predictions = model.transform(feature_df)

    query = predictions.writeStream.outputMode("append").format("console").start()
    query.awaitTermination()


if __name__ == "__main__":
    run_spark_streaming_example()
