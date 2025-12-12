import logging
import pandas as pd
from utils.config import RAW_DIR, PROCESSED_DIR
from .text_cleaning import clean_text

logger = logging.getLogger(__name__)


def clean_csv(
    input_filename: str,
    preserve_paragraphs: bool = False,
    do_lemmatize: bool = True,
) -> pd.DataFrame:
    """
    Clean and lemmatize CSV with text data.
    """

    input_path = RAW_DIR / input_filename
    output_path = PROCESSED_DIR / "clean_data.csv"

    df = pd.read_csv(input_path)
    df = df.dropna(subset=["text"])

    df["text"] = df["text"].apply(
        lambda x: clean_text(x, lemmatize=do_lemmatize)
    )

    if "title" in df.columns:
        df["title"] = df["title"].fillna("").apply(
            lambda x: clean_text(x, lemmatize=do_lemmatize)
        )

    df = df[df["text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["text"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Cleaned CSV saved to {output_path}")
    logger.info(f"Total rows: {len(df)}")

    return df
