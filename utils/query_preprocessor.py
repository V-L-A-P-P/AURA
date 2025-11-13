"""
query_preprocessor.py — модуль предобработки пользовательских запросов
для повышения точности поиска и извлечения релевантных документов.
"""

import re
import logging
import inspect
from functools import lru_cache
from typing import Optional, List

# 🩹 Совместимость со старыми библиотеками (Python ≥ 3.11)
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

# --- Совместимость pymorphy2 / pymorphy3 ---
try:
    import pymorphy2 as pymorphy
except ImportError:
    import pymorphy3 as pymorphy

from transformers import pipeline

logger = logging.getLogger(__name__)


class QueryPreprocessor:
    def __init__(
        self,
        use_llm: bool = False,
        llm_model: Optional[str] = None,
        enable_lemmatization: bool = True,
        enable_spellcheck: bool = False,
        cache_size: int = 1000
    ):
        """
        Конфигурация препроцессора запросов.
        """
        self.enable_lemmatization = enable_lemmatization
        self.enable_spellcheck = enable_spellcheck

        # --- Лемматизатор ---
        self._lemmatize_cached = lambda w: w  # безопасный fallback
        self.morph = None
        if enable_lemmatization:
            try:
                self.morph = pymorphy.MorphAnalyzer()
                self._lemmatize_cached = lru_cache(maxsize=cache_size)(self._lemmatize_word)
            except Exception as e:
                logger.warning(f"⚠️ Ошибка инициализации лемматизатора: {e}")
                self.morph = None

        # --- LLM для переформулировки ---
        self.llm_pipeline = None
        if use_llm and llm_model:
            try:
                logger.info(f"🔍 Загружаю LLM: {llm_model}")
                # Определяем тип модели (T5 → seq2seq, Llama/Mistral → text-generation)
                if any(x in llm_model.lower() for x in ["t5", "mt5", "mbart"]):
                    task = "text2text-generation"
                else:
                    task = "text-generation"

                self.llm_pipeline = pipeline(
                    task,
                    model=llm_model,
                    tokenizer=llm_model,
                    max_new_tokens=64,
                    do_sample=False
                )
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки LLM ({llm_model}): {e}")
                self.llm_pipeline = None

        # --- Синонимы и паттерны ---
        self.synonyms = self._load_synonyms()
        self.clean_patterns = [
            (r"<[^>]+>", " "),
            (r"[^а-яa-z0-9\s\-\.?!]", " "),
            (r"\s+", " "),
        ]

    # ====== Основные утилиты ====== #

    def _load_synonyms(self) -> dict:
        """Мини-словарь тематических синонимов"""
        return {
            "карта": ["карточка", "банковская карта", "кредитка"],
            "кредитная карта": ["кредитка", "кредитная карточка"],
            "дебетовая карта": ["дебетовая карточка", "расчетная карта"],
            "счет": ["банковский счет", "аккаунт"],
            "вклад": ["депозит", "сбережения"],
            "кредит": ["заем", "ссуда"],
            "ипотека": ["ипотечный кредит"],
            "перевод": ["транзакция", "платеж"],
            "ошибка": ["сбой", "проблема"],
            "приложение": ["мобильное приложение", "банковское приложение"],
            "вход": ["авторизация", "логин"],
            "пароль": ["пин-код", "код доступа"],
        }

    def _lemmatize_word(self, word: str) -> str:
        """Лемматизация одного слова"""
        if not self.morph:
            return word
        if len(word) <= 2 or word.isdigit():
            return word
        try:
            return self.morph.parse(word)[0].normal_form
        except Exception:
            return word

    def clean_text(self, text: str) -> str:
        """Очистка текста от мусора"""
        if not text:
            return ""
        text = text.lower().strip()
        for pattern, repl in self.clean_patterns:
            text = re.sub(pattern, repl, text)
        return text.strip()

    def lemmatize(self, text: str) -> str:
        """Лемматизация всего текста"""
        if not self.enable_lemmatization or not text:
            return text
        return " ".join(self._lemmatize_cached(w) for w in text.split())

    def expand_synonyms(self, text: str, max_expansions: int = 2) -> str:
        """Добавляет релевантные синонимы, сохраняя порядок"""
        words = text.split()
        expanded = []

        for i, word in enumerate(words):
            expanded.append(word)
            if word in self.synonyms:
                expanded.extend(self.synonyms[word][:max_expansions])
            elif i < len(words) - 1:
                phrase = f"{word} {words[i+1]}"
                if phrase in self.synonyms:
                    expanded.extend(self.synonyms[phrase][:max_expansions])

        cleaned = []
        for token in expanded:
            if not cleaned or cleaned[-1] != token:
                cleaned.append(token)
        return " ".join(cleaned)

    def rewrite_with_llm(self, text: str) -> str:
        """Переформулировка через LLM"""
        if not self.llm_pipeline or len(text.split()) > 20:
            return text
        try:
            prompt = (
                f"""
    Ты — интеллектуальный ассистент, который помогает системе поиска находить релевантные документы в базе знаний финтех компании.  
    Твоя задача — переформулировать исходный запрос, сохранив его исходный смысл, но изменяя стиль, лексику, цель запроса, структуру и способ выражения мысли.  

    Каждый вариант должен звучать естественно, как если бы его задал другой человек.  
    Не добавляй новые факты или уточнения — только меняй форму. ❗️ Не теряй важную информацию.

    ---

    ### Примеры (few-shot)

    Пример 1  
    Исходный запрос: "Как оплатить кредит?"  
    Перефразированные версии:  
    - "Какие способы оплаты кредита доступны?"  
    - "Как можно внести платеж по кредиту?"  
    - "Каким образом оплатить кредитный долг?"  

    Пример 2  
    Исходный запрос: "Проблемы с входом в личный кабинет"  
    Перефразированные версии:  
    - "Не получается войти в личный кабинет"  
    - "Ошибка при попытке входа в личный кабинет"  
    - "Почему не удается авторизоваться в личном кабинете?"  

    Пример 3  
    Исходный запрос: "Как получить карту?"  
    Перефразированные версии:  
    - "Что нужно, чтобы оформить карту?"  
    - "Как оформить и получить банковскую карту?"  
    - "Каким образом можно заказать карту?"  

    ---

    Теперь обработай следующий запрос:

    Исходный запрос: {text}

    Сгенерируй {3} перефразированных версий.  
    Каждый вариант — новая строка без нумерации.
    """
                )

            result = self.llm_pipeline(prompt)[0]["generated_text"]
            if "Переформулированный вариант" in result:
                result = result.split("Переформулированный вариант")[-1].strip()
            return result.strip() if result else text
        except Exception as e:
            logger.warning(f"⚠️ Ошибка LLM при переформулировке: {e}")
            return text

    def process(self, text: str, use_llm: Optional[bool] = None) -> str:
        """Полный конвейер предобработки"""
        if not text or not text.strip():
            return ""

        logger.info(f"🎯 Исходный запрос: {text}")
        cleaned = self.clean_text(text)
        lemmatized = self.lemmatize(cleaned)
        expanded = self.expand_synonyms(lemmatized)

        should_use_llm = use_llm if use_llm is not None else self.llm_pipeline is not None
        final = self.rewrite_with_llm(expanded) if should_use_llm else expanded

        logger.info(f"✨ Обработанный запрос: {final}")
        return final


# ===== Утилита для быстрой инициализации ===== #
def create_default_preprocessor() -> QueryPreprocessor:
    return QueryPreprocessor(
        use_llm=True,
        #llm_model="unsloth/Llama-3.2-3B-Instruct",
        #llm_model="Qwen/Qwen2.5-7B-Instruct",
        llm_model="Qwen/Qwen2.5-1.5B-Instruct",

        enable_lemmatization=True
    )


if __name__ == "__main__":
    pre = create_default_preprocessor()
    test_queries = [
        "Пропала кредитная карта из моего приложения",
        "Не могу войти в интернет банк",
        "Ошибка при переводе средств"
    ]
    for q in test_queries:
        print(f"\n🔍 Исходный: {q}")
        print(f"✨ Обработанный: {pre.process(q)}")
