"""
simulator.py
------------
Real-time transaction simulator (Kafka NOT required). Reads the processed
dataset, sends one transaction at a time through the prediction + risk +
alert pipeline, with a configurable delay to mimic a live feed.

Run: python -m streaming.simulator
Run: python -m streaming.simulator --limit 20 --delay 0.5
"""
from __future__ import annotations

import argparse
import time
import pandas as pd

from src import config
from src.realtime import process_transaction
from src.utils import get_logger

logger = get_logger(__name__)


def load_stream_source() -> pd.DataFrame:
    if config.PROCESSED_DATASET_PATH.exists():
        return pd.read_csv(config.PROCESSED_DATASET_PATH)
    return pd.read_csv(config.RAW_DATASET_PATH)


def run_simulation(limit: int = 50, delay: float = None) -> None:
    delay = config.SIMULATOR_DELAY_SECONDS if delay is None else delay
    df = load_stream_source().sample(frac=1, random_state=None).head(limit)

    logger.info(f"Starting real-time simulation: {len(df)} transactions, {delay}s delay")

    for _, row in df.iterrows():
        transaction = row.to_dict()
        try:
            result = process_transaction(transaction)
            print(
                f"TXN {result['transaction_id']:>10} | "
                f"prob={result['fraud_probability']:.3f} | "
                f"score={result['risk_score']:>3} | "
                f"level={result['risk_level']}"
            )
        except Exception as exc:
            logger.error(f"Failed to process transaction: {exc}")
        time.sleep(delay)

    logger.info("Simulation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time fraud transaction simulator")
    parser.add_argument("--limit", type=int, default=50, help="Number of transactions to simulate")
    parser.add_argument("--delay", type=float, default=None, help="Delay in seconds between transactions")
    args = parser.parse_args()
    run_simulation(limit=args.limit, delay=args.delay)
