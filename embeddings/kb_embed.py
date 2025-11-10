"""
kb_embed.py — создание эмбеддингов и FAISS-индекса для базы знаний (knowledge base).

Этапы:
 1. Загрузка текстовых чанков из data/chunks/
 2. Кодирование текстов через SentenceTransformer
 3. Сохранение векторов, метаданных и индекса FAISS
"""
from pathlib import Path
import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer
from utils.config import CHUNKS_DIR, EMBEDDINGS_DIR, EMBEDDING_MODEL

# ==========================
# 📂 Загрузка чанков
# ==========================

def load_text_chunks(chunks_dir: Path = CHUNKS_DIR):
    """
    Загружает все .txt-файлы из data/chunks и возвращает список текстов и метаданные.
    """
    texts, meta = [], []

    if not chunks_dir.exists():
        raise FileNotFoundError(f"❌ Папка с чанками не найдена: {chunks_dir}")

    for file in sorted(chunks_dir.glob("*.txt")):
        text = file.read_text(encoding="utf-8").strip()
        if len(text) < 20:
            continue
        texts.append(text)
        meta.append({"file": file.name})

    if not texts:
        raise ValueError("❌ Не найдено чанков для обработки. Сначала запустите data/pipeline.py.")

    print(f"📄 Загружено {len(texts)} чанков из {chunks_dir}")
    return texts, meta


# ==========================
# 🧠 Создание эмбеддингов
# ==========================

def build_kb_embeddings(model_name: str = EMBEDDING_MODEL):
    """
    Кодирует все чанки базы знаний и создаёт FAISS-индекс.
    Результаты сохраняются в embeddings/:
      - kb_vectors.npy
      - kb_metadata.json
      - kb_index.faiss
    """
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    texts, meta = load_text_chunks(CHUNKS_DIR)

    print(f"🧠 Загружаем модель эмбеддингов: {model_name}")
    model = SentenceTransformer(model_name)

    print("🔢 Кодируем чанки...")
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    np.save(EMBEDDINGS_DIR / "kb_vectors.npy", vectors)
    with open(EMBEDDINGS_DIR / "kb_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Эмбеддинги сохранены: {vectors.shape[0]} векторов размерности {vectors.shape[1]}")

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    faiss.write_index(index, str(EMBEDDINGS_DIR / "kb_index.faiss"))

    print(f"📦 FAISS-индекс сохранён в {EMBEDDINGS_DIR / 'kb_index.faiss'}")


# ==========================
# 🧪 Тестовый запуск
# ==========================

if __name__ == "__main__":
    """
    Пример самостоятельного запуска:
        python -m embeddings.kb_embed
    """
    try:
        build_kb_embeddings()
    except Exception as e:
        print(f"❌ Ошибка при создании эмбеддингов базы знаний: {e}")
