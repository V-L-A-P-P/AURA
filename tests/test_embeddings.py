import os
import json
import numpy as np
import faiss
import pytest
from pathlib import Path

from embeddings.websites_embed import build_kb_embeddings
from utils.config import EMBEDDINGS_DIR

def test_build_kb_embeddings_creates_files(tmp_path, monkeypatch):
    """
    Проверяет, что после запуска build_kb_embeddings появляются:
      - файлы векторов (.npy)
      - метаданные (.json)
      - индекс (.faiss)
    А также что размерность индекса соответствует размерности векторов.
    """

    # Подменим EMBEDDINGS_DIR на tmp_path для изоляции
    monkeypatch.setenv("EMBEDDINGS_DIR", str(tmp_path))
    # Также можно подменить конфиг внутри модуля:
    from importlib import reload
    import utils.config
    reload(utils.config)

    # Запускаем функцию (с batch_size=1 для ускорения теста)
    build_kb_embeddings(batch_size=1)

    # Проверяем файлы
    vec_file  = tmp_path / "kb_vectors.npy"
    meta_file = tmp_path / "kb_metadata.json"
    idx_file  = tmp_path / "kb_index.faiss"

    assert vec_file.exists(),  "Файл векторов не создан"
    assert meta_file.exists(), "Файл метаданных не создан"
    assert idx_file.exists(),  "Файл индекса FAISS не создан"

    # Загружаем векторы
    vectors = np.load(vec_file)
    assert vectors.ndim == 2
    n, d = vectors.shape
    assert n > 0 and d > 0, "Неправильная форма векторов"

    # Загружаем индекс и проверяем размерность
    index = faiss.read_index(str(idx_file))
    # IndexFlatL2 хранит параметр d
    assert index.d == d, f"Размерность индекса ({index.d}) != размерность векторов ({d})"

    # Загружаем метаданные и проверяем количество записей = n
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert len(metadata) == n, "Количество метаданных не совпадает с количеством векторов"

    # Проверка ключей метаданных у первой записи
    first = metadata[0]
    for key in ("web_id", "chunk_id", "title", "url", "text"):
        assert key in first, f"В метаданных отсутствует ключ «{key}»"

    # Дополнительная простая проверка: поиск ближнего соседа для первого вектора
    D, I = index.search(vectors[:1], k=1)
    assert I[0][0] == 0, "Первый вектор должен находить себя"

if __name__ == "__main__":
    pytest.main([__file__])
