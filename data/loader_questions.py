"""
loader_questions.py — загрузка и предварительная обработка файла с вопросами.
Формирует файл processed/questions.json или csv-формат, готовый к дальнейшему использованию.
"""

import pandas as pd
from pathlib import Path
import json
from utils.config import RAW_DIR, PROCESSED_DIR

def load_questions(csv_path: Path = None) -> pd.DataFrame:
    """
    Загружает CSV с вопросами.
    Ожидается наличие столбцов: q_id, query
    """
    if csv_path is None:
        csv_path = RAW_DIR / "questions_clean_sample_200.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"❌ Файл с вопросами не найден: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)
    required = ["q_id", "query"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"❌ В файле вопросов отсутствует столбец «{col}»")
    df = df.dropna(subset=["q_id", "query"]).drop_duplicates(subset=["q_id"])
    df["q_id"] = df["q_id"].astype(str).str.strip()
    df["query"] = df["query"].astype(str).str.strip()
    print(f"🔹 Загружено {len(df)} вопросов.")
    return df

def save_processed_questions(df: pd.DataFrame, output_path: Path = None) -> None:
    """
    Сохраняет обработанный датасет вопросов в JSON-файл.
    """
    if output_path is None:
        output_path = PROCESSED_DIR / "questions.json"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    records = df.to_dict(orient="records")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"✅ Сохранены обработанные вопросы: {output_path}")

if __name__ == "__main__":
    df_q = load_questions()
    save_processed_questions(df_q)
