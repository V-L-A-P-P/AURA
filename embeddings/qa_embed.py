"""
qa_embed.py — создание эмбеддингов и FAISS-индекса для базы вопросов-ответов (QA).

Поддерживаются два режима:
 1. mode="questions" — индексируются только вопросы.
 2. mode="combined"  — индексируется текст “вопрос + ответ”.

Результаты:
  - qa_vectors_q.npy             (режим questions)
  - qa_questions_index.faiss
  - qa_vectors_combined.npy      (режим combined)
  - qa_combined_index.faiss
  - qa_metadata.json            (всегда)
"""

from pathlib import Path
import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer
from utils.config import PROCESSED_DIR, EMBEDDINGS_DIR, EMBEDDING_MODEL


def load_qa_data(qa_path: Path = None):
    """
    Загружает QA-датасет из JSON файла.
    """
    if qa_path is None:
        qa_path = PROCESSED_DIR / "qa_dataset.json"
    if not qa_path.exists():
        raise FileNotFoundError(f"❌ QA-файл не найден: {qa_path}")
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    print(f"🔹 Загружено {len(qa_data)} QA-записей.")
    return qa_data


def build_qa_index(mode: str = "questions", batch_size: int = 16, model_name: str = EMBEDDING_MODEL):
    """
    Строит эмбеддинги и FAISS-индекс для QA-базы.

    Параметры:
      mode        — «questions» или «combined»
      batch_size  — размер батча кодирования
      model_name  — имя модели SentenceTransformer
    """
    if mode not in ("questions", "combined"):
        raise ValueError("mode должен быть либо 'questions', либо 'combined'")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    qa_data = load_qa_data()

    if mode == "questions":
        texts = [rec.get("question", "").strip() for rec in qa_data]
        vectors_file = "qa_vectors_q.npy"
        index_file = "qa_questions_index.faiss"
    else:
        texts = [(rec.get("question", "").strip() + " " + rec.get("answer", "").strip())
                 for rec in qa_data]
        vectors_file = "qa_vectors_combined.npy"
        index_file = "qa_combined_index.faiss"

    if not texts or all(len(t) == 0 for t in texts):
        raise ValueError(f"❌ Нет текстов для кодирования (mode={mode}).")

    print(f"🧠 Загружаем модель эмбеддингов: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"🔢 Кодируем {mode} текстов с batch_size={batch_size} …")
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    vectors = vectors.astype('float32')

    # Сохранение
    np.save(EMBEDDINGS_DIR / vectors_file, vectors)
    with open(EMBEDDINGS_DIR / "qa_metadata.json", "w", encoding="utf-8") as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)
    print(f"✅ QA-метаданные сохранены: {len(qa_data)} записей.")
    print(f"✅ Эмбеддинги сохранены: {vectors.shape[0]} векторов размерности {vectors.shape[1]}")

    # Построение индекса
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    index_path = EMBEDDINGS_DIR / index_file
    faiss.write_index(index, str(index_path))
    print(f"📦 FAISS-индекс (mode={mode}) сохранён: {index_path}")


if __name__ == "__main__":
    """
    Пример запуска:
      python -m embeddings.qa_embed --mode combined
    """
    try:
        # Пример: режим "combined"
        build_qa_index(mode="combined", batch_size=16)
    except Exception as e:
        print(f"❌ Ошибка при создании QA-индекса: {e}")
