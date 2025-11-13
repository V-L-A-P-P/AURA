"""
chunk_utils.py — модуль для разбивки очищенных текстов на чанки.
Используется после очистки/предобработки, перед построением эмбеддингов.
"""

from pathlib import Path
import logging
from typing import List, Union
from utils.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNKS_DIR
import re
import nltk
from typing import List
from utils.config import CHUNK_SIZE, CHUNK_OVERLAP
from utils.config import RAW_DIR, PROCESSED_DIR, CHUNKS_DIR

# Скачиваем необходимые данные NLTK
import re
from typing import List
from utils.config import CHUNK_SIZE, CHUNK_OVERLAP
import logging

logger = logging.getLogger(__name__)

def is_noisy_chunk(text: str) -> bool:
    # если более 40% токенов — цифры, пунктуация или короткие слова
    tokens = re.findall(r'\w+', text)
    if not tokens:
        return True
    noise = sum(1 for t in tokens if re.fullmatch(r'\d+|\d+\.\d+|п|№|таблица|приложение', t.lower()))
    return noise / len(tokens) > 0.4

def split_text_to_chunks_advanced(
    text: str,
    max_words: int = CHUNK_SIZE,
    overlap_words: int = CHUNK_OVERLAP,
    min_words: int = None
) -> List[str]:
    """
    Быстрая и устойчивая разбивка текста на чанки по предложениям без NLTK.
    Встроена фильтрация шумных чанков (цифры, пункты, приложения и т.п.).
    """
    if not text or not text.strip():
        return []

    # 💨 Очистка мусора и нормализация пробелов
    text = re.sub(r"(©|cookie|javascript|обратная связь|контакты|все права защищены).*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    # ⚡ Быстрая разбивка на предложения
    sentence_endings = r'(?<=[.!?…])\s+(?=[А-ЯA-Z"(\[])'
    sentences = re.split(sentence_endings, text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Fallback, если regex не дал результата
    if not sentences or len(sentences) == 1:
        sentences = []
        buffer = ""
        for i, char in enumerate(text):
            buffer += char
            if char in ".!?…" and i < len(text) - 1 and text[i + 1] in " \n\t":
                sentences.append(buffer.strip())
                buffer = ""
        if buffer.strip():
            sentences.append(buffer.strip())

    if not sentences:
        return _split_by_words_fallback(text, max_words, overlap_words)

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)

        if word_count < 2 and len(sentence) < 10:
            if current_chunk:
                current_chunk[-1] += " " + sentence
                current_word_count += word_count
            continue

        if word_count > max_words * 0.8:
            if current_chunk:
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text and not is_noisy_chunk(chunk_text):
                    chunks.append(chunk_text)
                current_chunk = []
                current_word_count = 0

            sub_chunks = _split_large_sentence(sentence, max_words, overlap_words)
            for sub in sub_chunks:
                if sub and not is_noisy_chunk(sub):
                    chunks.append(sub)
            continue

        if current_word_count + word_count > max_words and current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text and not is_noisy_chunk(chunk_text):
                chunks.append(chunk_text)

            if overlap_words > 0:
                overlap_text = _create_overlap(current_chunk, overlap_words)
                current_chunk = [overlap_text] if overlap_text else []
                current_word_count = len(overlap_text.split()) if overlap_text else 0
            else:
                current_chunk = []
                current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += word_count

    # Добавляем последний чанк (с проверкой на шум)
    if current_chunk:
        chunk_text = " ".join(current_chunk).strip()
        if chunk_text and not is_noisy_chunk(chunk_text):
            chunks.append(chunk_text)

    if min_words is None:
        min_words = max(5, max_words // 4)

    chunks = [chunk for chunk in chunks if len(chunk.split()) >= min_words]

    if not chunks:
        return [text.strip()]

    avg_len = sum(len(c.split()) for c in chunks) / len(chunks)
    logger.debug(f"Создано {len(chunks)} чанков, средняя длина {avg_len:.1f} слов")

    return chunks


# ===== Вспомогательные функции ===== #

def _split_large_sentence(sentence: str, max_words: int, overlap_words: int) -> List[str]:
    """Разбивает очень большие предложения на подчанки."""
    words = sentence.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap_words
    return chunks


def _create_overlap(chunk_sentences: List[str], overlap_words: int) -> str:
    """Создаёт перекрытие из последних слов чанка."""
    if not chunk_sentences:
        return ""
    full_text = " ".join(chunk_sentences)
    words = full_text.split()
    if len(words) <= overlap_words:
        return full_text
    overlap_text = " ".join(words[-overlap_words:])
    return re.sub(r"\s+", " ", overlap_text).strip()


def _split_by_words_fallback(text: str, max_words: int, overlap_words: int) -> List[str]:
    """Fallback: простой чанкинг по количеству слов."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap_words
    return chunks


from sentence_transformers import SentenceTransformer, util
import nltk
import numpy as np
nltk.download('punkt')

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def semantic_chunk(
        text,
        max_words=600,
        min_words=200,
        overlap_words=120,
        sim_threshold=0.75,
        centroid_smoothing=0.2
    ):

    sentences = nltk.sent_tokenize(text)
    embeddings = model.encode(sentences, normalize_embeddings=True).astype(np.float32)

    chunks = []
    current_chunk = []
    current_embeds = []
    current_len = 0

    def get_centroid():
        if not current_embeds:
            return None
        return np.mean(np.stack(current_embeds, axis=0), axis=0)

    for sent, emb in zip(sentences, embeddings):

        if current_chunk:
            centroid = get_centroid()

            # сглаживание центроида
            centroid = centroid * (1 - centroid_smoothing) + emb * centroid_smoothing

            sim = util.cos_sim(emb, centroid).item()

            too_long = (current_len + len(sent.split()) > max_words)
            topic_shift = (sim < sim_threshold and current_len >= min_words)

            if too_long or topic_shift:

                chunks.append(" ".join(current_chunk).strip())

                # --- защитный overlap ---
                if overlap_words > 0 and len(current_chunk) > 1:

                    words_kept = 0
                    overlap_chunk = []
                    overlap_embeds = []

                    for s, e in zip(reversed(current_chunk), reversed(current_embeds)):
                        w = len(s.split())
                        overlap_chunk.insert(0, s)
                        overlap_embeds.insert(0, e)
                        words_kept += w
                        if words_kept >= overlap_words:
                            break

                    current_chunk = overlap_chunk
                    current_embeds = overlap_embeds
                    current_len = words_kept

                else:
                    current_chunk = []
                    current_embeds = []
                    current_len = 0

        # добавляем текущее предложение
        current_chunk.append(sent)
        current_embeds.append(emb)
        current_len += len(sent.split())

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks



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
            chunks = split_text_to_chunks(text, max_words=chunk_size, overlap_words=overlap)
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


import os
import re
import pandas as pd
import pymorphy3


morph = pymorphy3.MorphAnalyzer()


def lemmatize_text(text: str) -> str:
    """Лемматизация текста для русского языка"""
    words = text.split()
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    return " ".join(lemmas)


def clean_text(text: str, preserve_paragraphs: bool = False, do_lemmatize: bool = True) -> str:
    """Очистка текста + лемматизация"""
    if not isinstance(text, str):
        return ""

    # Замены спецсимволов
    text = text.replace("\xa0", " ")
    text = text.replace("₽", "рублей")
    text = text.replace("$", "долларов")
    text = text.replace("→", "-")
    text = text.replace('дм³', 'дм³ (кубических дециметров)')

    # Убираем длинные последовательности точек, тире, подчёркиваний
    text = re.sub(r"(\.{3,}|_{2,}|-{3,})", " ", text)

    # Мусорные символы
    text = re.sub(r"[«»\"“”‘’\[\]{}<>|•■◆►]", " ", text)

    # Абзацы
    if preserve_paragraphs:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
    else:
        text = re.sub(r"\s+", " ", text)

    # Пробелы перед пунктуацией
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r" {2,}", " ", text)

    # Нижний регистр
    text = text.lower()

    # Лемматизация
    if do_lemmatize:
        text = lemmatize_text(text)

    return text.strip()


def clean_csv(input_filename: str, preserve_paragraphs: bool = False, do_lemmatize: bool = True):
    """Очистка CSV с текстами, лемматизация и сохранение"""
    input_path = RAW_DIR / input_filename
    output_path = PROCESSED_DIR / "clean_data.csv"

    data = pd.read_csv(input_path)
    data = data.dropna(subset=["text"])

    data["text"] = data["text"].apply(lambda x: clean_text(x, preserve_paragraphs, do_lemmatize))
    data["title"] = data.get("title", pd.Series([""] * len(data))).apply(
        lambda x: clean_text(x, preserve_paragraphs, do_lemmatize)
    )

    # Убираем пустые строки и дубли
    data = data[data["text"].str.strip() != ""].drop_duplicates(subset=["text"])

    data.to_csv(output_path, index=False)

    print(f"Cleaned data saved in: {output_path}")
    print(f"Всего записей: {len(data)}")

    return data

if __name__ == "__main__":
    clean_csv('websites_updated.csv', True)