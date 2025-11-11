"""
chunk_utils.py — модуль для разбивки очищенных текстов на чанки.
Используется после очистки/предобработки, перед построением эмбеддингов.
"""

from pathlib import Path
import logging
from typing import List, Union
from utils.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNKS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_text_to_chunks(
    text: str,
    max_words: int = CHUNK_SIZE,
    overlap_words: int = CHUNK_OVERLAP,
    min_words: int = None
) -> List[str]:
    """
    Разбивает текст на чанки по словам:
      - max_words      : максимальное число слов в чанке
      - overlap_words  : перекрытие между чанками
      - min_words      : минимальное число слов для разбивки; если текст короче — возвращается как единый чанок
    Возвращает список чанков (каждый — строка).
    """
    if max_words <= 0:
        raise ValueError("CHUNK_SIZE должен быть > 0")
    if overlap_words >= max_words:
        raise ValueError("CHUNK_OVERLAP должен быть меньше CHUNK_SIZE")

    words = text.split()
    length = len(words)
    if min_words is None:
        min_words = max_words // 2
    if length <= min_words:
        logger.debug(f"Текст слишком короткий ({length} слов) — возвращаю весь как один чанок.")
        return [text.strip()]

    chunks: List[str] = []
    start = 0
    while start < length:
        end = min(start + max_words, length)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start += (max_words - overlap_words)

    logger.info(f"Разбито на {len(chunks)} чанков ({length} слов → max_words={max_words}, overlap={overlap_words})")
    return chunks


def save_chunks_from_files(
    input_dir: Union[Path, str],
    output_dir: Union[Path, str] = CHUNKS_DIR,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> None:
    """
    Читает все .txt файлы из input_dir, разбивает каждый на чанки и
    сохраняет их в output_dir как отдельные файлы.
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_path.glob("*.txt"))
    if not txt_files:
        logger.warning(f"⚠️ В папке {input_path} нет файлов для разбиения.")
        return

    logger.info(f"✂️ Найдено {len(txt_files)} файлов для разбивки из {input_path}")
    for file_path in txt_files:
        try:
            text = file_path.read_text(encoding="utf-8")
            chunks = create_chunks_from_text(text, chunk_size, overlap)
            if not chunks:
                logger.warning(f"⚠️ {file_path.name}: не удалось создать чанки (пустой текст).")
                continue
            base_name = file_path.stem.replace(" ", "_")
            for i, chunk in enumerate(chunks, start=1):
                chunk_filename = f"{base_name}_chunk_{i}.txt"
                chunk_file = output_path / chunk_filename
                chunk_file.write_text(chunk, encoding="utf-8")
            logger.info(f"✅ {file_path.name}: создано {len(chunks)} чанков")
        except Exception as e:
            logger.error(f"❌ Ошибка при обработке {file_path.name}: {e}")

    logger.info(f"📂 Разбиение завершено. Проверьте папку: {output_path.resolve()}")
