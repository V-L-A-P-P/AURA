# AURA — Augmented Understanding & Retrieval for Answers

**AURA** — production-oriented RAG-система для поиска и генерации ответов по финтех-корпусу документов.
Проект реализует **полный end-to-end пайплайн**: от очистки данных и построения индексов до генерации финального ответа LLM с указанием источников.

Ключевой фокус — **качество retrieval**, превосходящее базовый RAG за счёт гибридного поиска, graph-augmented расширения кандидатов, LLM-driven query rewriting и transformer-based reranking.

---

## 🚀 Ключевые возможности

### Hybrid Retrieval

**Поиск и ранжирование кандидатов:**

* TF-IDF (lexical search)
* Dense retrieval через FAISS
* Двухэтапный transformer-based cross-encoder reranking

---

### Graph-Augmented Retrieval

**Работа со структурой документа:**

* Граф связей между чанками документов
* Расширение кандидатов через граф
* Повышение recall и устойчивости поиска

---

### Гибкая подготовка данных

**Предобработка и чанкинг:**

* Очистка и нормализация текстов
* Несколько стратегий чанкинга:

  * базовый (по словам)
  * семантический
  * рекурсивный

---

### LLM-driven Query Rewriting

**Улучшение пользовательского запроса перед retrieval:**

* Перефразирование пользовательского запроса перед retrieval
* Используется instruction-tuned LLM
* Повышает полноту и устойчивость поиска при коротких или неявных запросах

---

### LLM Answer Generation

**Формирование финального ответа:**

* Генерация ответа строго на основе найденных документов
* Явное указание источников (чанков)
* Защита от hallucinations

---

### Production-ready архитектура

**Инженерная готовность к использованию:**

* Чёткое разделение offline / online этапов
* Тестируемые компоненты (PyTest)
* API для онлайн-инференса (FastAPI)

---

## 🧠 Архитектура пайплайна

```text
User Query
    ↓
LLM Query Rewriting
    ↓
Hybrid Retrieval (TF-IDF + FAISS)
    ↓
Graph-based Expansion
    ↓
Transformer Reranking
    ↓
Top-K Relevant Chunks
    ↓
LLM Answer Generation (with sources)
```

---

## 📂 Структура проекта

```text
AURA/
├── chunking/
│   ├── base.py
│   ├── semantic.py
│   ├── recursive.py
│   └── filters.py
│
├── preprocessing/
│   ├── csv_cleaner.py
│   └── text_cleaning.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── loader_websites.py
│   └── loader_questions.py
│
├── embeddings/
│   ├── websites_embed.py
│   ├── kb_vectors.npy
│   ├── kb_index.faiss
│   ├── kb_metadata.json
│   └── tfidf_model.pkl
│
├── graph/
│   ├── build_graph.py
│   ├── graph_expander.py
│   ├── graph_store.py
│   └── graph_index.pkl
│
├── retrieval/
│   ├── retriever.py
│   └── pipeline_check.py
│
├── llm/
│   ├── query_expander.py
│   ├── answer_generator.py
│   └── marked_llm.py
│
├── api/
│   └── main.py
│
├── tests/
│   ├── test_embeddings.py
│   ├── test_retrieval.py
│   └── test_generation.py
│
├── run_pipeline.py
├── rag_answer.py
├── pipeline.py
│
├── utils/
│   └── config.py
│
├── requirements.txt
├── environment_full.yml
└── aura2.md
```

---

## ⚙️ Запуск пайплайна

### 1️⃣ Полный offline-пайплайн

```bash
python run_pipeline.py
```

**Этапы выполнения:**

* загрузка и очистка документов
* чанкинг (несколько стратегий)
* построение эмбеддингов и FAISS-индекса
* построение графа связей между чанками
* hybrid retrieval по списку вопросов

---

### 2️⃣ Генерация ответа (локально)

```bash
python rag_answer.py
```

```python
from rag_answer import answer_query

result = answer_query(
    "Как оплатить кредит через приложение?",
    top_k=5
)

print(result["answer"])
```

---

## 🌐 API (FastAPI)

### Запуск сервиса

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Пример запроса

```http
POST /answer
Content-Type: application/json

{
  "query": "Как оплатить кредит через мобильное приложение банка?",
  "top_k": 5
}
```

**API возвращает:**

* исходный вопрос
* найденные релевантные чанки
* финальный ответ LLM с указанием источников

---

## 📘 Пример работы системы

### Вопрос

Как оплатить кредит через мобильное приложение банка?

### Найденные документы

```
[DOC 1]
В мобильном приложении банка доступна оплата кредитов с дебетовой карты,
а также настройка автоматического ежемесячного платежа.

[DOC 2]
В приложении можно посмотреть график платежей по кредиту и сумму
обязательного платежа на текущий месяц.

[DOC 3]
Документ описывает условия оформления нового кредита.
```

### Ответ

```
Оплатить кредит через мобильное приложение банка можно с помощью встроенной
функции оплаты кредитов. В приложении доступен перевод средств с дебетовой
карты на кредитный счёт, а также возможность настроить автоматический платёж
на нужную дату [DOC 1].

Дополнительно в приложении можно проверить график платежей и сумму
обязательного взноса за текущий месяц, чтобы избежать просрочек [DOC 2].
```

### Обоснование

```
[DOC 1]: Relevant — содержит прямую информацию о способах оплаты кредита
через мобильное приложение.

[DOC 2]: Relevant — дополняет ответ информацией о контроле платежей.

[DOC 3]: Not relevant — относится к оформлению кредита, а не к его оплате.
```

---

## 🧪 Тестирование

```bash
pytest
```

---

## 🛠 Стек технологий

Python, FAISS, Sentence-Transformers, Cross-Encoder, TF-IDF, NetworkX, LLM, FastAPI, PyTest

---

## 📌 Roadmap

* Метрики retrieval (Recall@K, MRR)
* Chat-интерфейс
* Streaming-ответы LLM
