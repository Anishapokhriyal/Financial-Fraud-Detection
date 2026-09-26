"""
kafka_producer.py
------------------
OPTIONAL advanced module. Streams transactions onto a Kafka topic.

This is NOT required to run the core project (see streaming/simulator.py for
the no-Kafka version). Install kafka-python and run a local Kafka broker to
use this module:
    pip install kafka-python
    # start Zookeeper + Kafka locally (see README "Optional Kafka" section)
    python -m streaming.kafka_producer

Fixes vs. original spec (which had typos: KfkaConsumer, KafkaComsumer,
json/loads, and a malformed bootstrap_servers list):
"""
from __future__ import annotations

import json
import time
import pandas as pd

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

TOPIC_NAME = "transaction_stream"


def run_producer(bootstrap_servers: str = "localhost:9092", delay: float = 1.0):
    try:
        from kafka import KafkaProducer
    except ImportError:
        raise ImportError(
            "kafka-python is not installed. Run: pip install kafka-python"
        )

    producer = KafkaProducer(
        bootstrap_servers=[bootstrap_servers],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    df = pd.read_csv(config.PROCESSED_DATASET_PATH)

    logger.info(f"Producing {len(df)} transactions to topic '{TOPIC_NAME}'")
    for _, row in df.iterrows():
        message = row.to_dict()
        producer.send(TOPIC_NAME, value=message)
        logger.info(f"Produced transaction {message.get('Transaction_ID')}")
        time.sleep(delay)

    producer.flush()
    producer.close()


if __name__ == "__main__":
    run_producer()
