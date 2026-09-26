"""
run.py
------
Convenience script that runs the full offline pipeline end-to-end:
ETL -> Feature Engineering -> Train Models -> Init DB -> Load Transactions.

This does NOT start the API or dashboard (run those separately, see README).

Run: python run.py
"""
from src.train_model import main as train_main
from src.database import init_database, load_transactions_from_dataframe
from src import config
from src.utils import get_logger
import pandas as pd

logger = get_logger(__name__)


def main():
    logger.info("STEP 1/3: Training model (ETL + feature engineering + training run inside)")
    train_main()

    logger.info("STEP 2/3: Initializing database")
    init_database()

    logger.info("STEP 3/3: Loading processed transactions into database")
    df = pd.read_csv(config.PROCESSED_DATASET_PATH)
    n = load_transactions_from_dataframe(df)
    from src.database import backfill_predictions_from_transactions
    predictions_generated = backfill_predictions_from_transactions()
    logger.info(f"Loaded {n} transactions and backfilled {predictions_generated} predictions")

    print("\nSetup complete.")
    print("Next steps:")
    print("  uvicorn api.main:app --reload      # start the API")
    print("  streamlit run dashboard/app.py     # start the dashboard")


if __name__ == "__main__":
    main()
