import os
import pandas as pd

from query_expander import QueryExpander
from tqdm import tqdm



RAW_DATA_DIR = "data/raw"
OUTPUT_DIR =  "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_queries(
    input_filename: str = "questions_clean.csv",
    output_filename: str = "questions_llm_augmented.csv",
    mode: str = "paraphrase",  # 'expand' или 'paraphrase'
    n_variants: int = 1,
):
    """
    Обрабатывает вопросы через LLM и сохраняет новые варианты.
    """
    input_path = os.path.join(RAW_DATA_DIR, input_filename)
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    df = pd.read_csv(input_path, nrows=6977)
    if "q_id" not in df.columns or "query" not in df.columns:
        raise ValueError("CSV должен содержать колонки: q_id, query")

    expander = QueryExpander()
    results = []

    print(f"Обработка {len(df)} запросов в режиме '{mode}' ...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        q_id = row["q_id"]
        query = str(row["query"]).strip()

        if len(query) > 1000:
            print(query)
            print('>50')

            results.append({
                "q_id": q_id,
                "query": query
            })
        else:

            try:
                variants = expander.generate(query, mode=mode, n_variants=n_variants)
            except Exception as e:
                print(f"Ошибка при обработке запроса {q_id}: {e}")
                variants = []

            results.append({
                "q_id": q_id,
                "query": variants[0]
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)

    print(f"Готово! Результаты сохранены в: {output_path}")


if __name__ == "__main__":
    process_queries(
        input_filename="questions_clean.csv",
        output_filename="questions_expanded.csv",
        mode="paraphrase",
        n_variants=1
    )
