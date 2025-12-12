"""
loader_questions.py — loading and preprocessing questions dataset.
Creates processed/questions.json for downstream retrieval.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from utils.config import RAW_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_questions(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Load questions from CSV.
    Required columns: q_id, query
    """
    if csv_path is None:
        csv_path = RAW_DIR / "questions_clean.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Questions file not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str)

    required_cols = {"q_id", "query"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = (
        df.dropna(subset=["q_id", "query"])
          .drop_duplicates(subset=["q_id"])
    )

    df["q_id"] = df["q_id"].astype(str).str.strip()
    df["query"] = df["query"].astype(str).str.strip()

    logger.info(f"Loaded {len(df)} questions")

    return df


def save_processed_questions(df: pd.DataFrame, output_path: Path | None = None) -> None:
    """
    Save processed questions to JSON.
    """
    if output_path is None:
        output_path = PROCESSED_DIR / "questions.json"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    logger.info(f"Processed questions saved to {output_path}")


if __name__ == "__main__":
    df_q = load_questions()
    save_processed_questions(df_q)
