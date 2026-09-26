"""
kafka_consumer.py
------------------
OPTIONAL advanced module. Consumes transactions from a Kafka topic and runs
them through the same prediction/risk/alert pipeline used by the simulator.

Install and run a local Kafka broker, then:
    pip install kafka-python
    python -m streaming.kafka_consumer

Fixes vs. original spec (KfkaConsumer/KafkaComsumer typos, json/loads typo,
malformed bootstrap_servers list are all corrected below).
"""
from __future__ import annotations

import json

from src.realtime import process_transaction
from src.utils import get_logger
from streaming.kafka_producer import TOPIC_NAME

logger = get_logger(__name__)


def run_consumer(bootstrap_servers: str = "localhost:9092"):
    try:
        from kafka import KafkaConsumer
    except ImportError:
        raise ImportError(
            "kafka-python is not installed. Run: pip install kafka-python"
        )

    consumer = KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=[bootstrap_servers],
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
    )

    logger.info(f"Listening on topic '{TOPIC_NAME}' ...")
    for message in consumer:
        transaction = message.value
        result = process_transaction(transaction)
        if result["is_fraud"] == 1:
            logger.warning(f"ALERT! Fraudulent transaction detected: {result}")
        else:
            logger.info(f"Transaction processed: {result}")


if __name__ == "__main__":
    run_consumer()
