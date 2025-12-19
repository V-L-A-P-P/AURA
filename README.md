# AURA — Augmented Understanding & Retrieval for Answers

**AURA** — production-oriented RAG-система для поиска и генерации ответов по финтех-корпусу документов.  
Проект реализует **полный end-to-end пайплайн**: от очистки данных и построения индексов до генерации финального ответа LLM с указанием источников.

Ключевой фокус — **качество retrieval**, превосходящее базовый RAG за счёт гибридного поиска, graph-augmented расширения кандидатов и transformer-based reranking.

---

## 🚀 Ключевые возможности

### Hybrid Retrieval
- TF-IDF (lexical search)
- Dense retrieval через FAISS
- Двухэтапный transformer-based cross-encoder reranking

### Graph-Augmented Retrieval
- Граф связей между чанками документов
- Расширение кандидатов через граф
- Повышение recall и устойчивости поиска

### Гибкая подготовка данных
- Очистка и нормализация текстов
- Несколько стратегий чанкинга:
  - базовый (по словам)
  - семантический
  - рекурсивный

### LLM-driven Query Rewriting
- Переформулировка пользовательских запросов
- Повышение полноты и релевантности retrieval

### LLM Answer Generation
- Генерация ответа строго на основе найденных документов
- Явное указание источников (чанков)
- Защита от hallucinations

### Production-ready архитектура
- Чёткое разделение offline / online этапов
- Тестируемые компоненты (PyTest)
- Подготовка к API и деплою

---

## 🧠 Архитектура пайплайна

```text
Raw CSV / Text Data
    ↓
Text Cleaning & Preprocessing
    ↓
Chunking (base / semantic / recursive)
    ↓
Embeddings + FAISS Index
    ↓
Chunk Graph Construction
    ↓
Hybrid Retrieval (TF-IDF + Dense)
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
└── aura.md
```

---

## ⚙️ Запуск пайплайна

### 1️⃣ Полный offline-пайплайн

```bash
python run_pipeline.py
```

Этапы:
- загрузка и очистка документов
- чанкинг (несколько стратегий)
- построение эмбеддингов и FAISS-индекса
- построение графа связей между чанками
- hybrid retrieval по списку вопросов

---

### 2️⃣ Только retrieval (если данные уже подготовлены)

```python
from run_pipeline import run_retrieval_only

run_retrieval_only()
```

---

## 🤖 Генерация ответа (RAG inference)

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

## 🧪 Тестирование

```bash
pytest
```

---

## 🛠 Стек технологий

Python, FAISS, Sentence-Transformers, Cross-Encoder, TF-IDF, NetworkX, LLM, FastAPI, PyTest

---

## 📌 Roadmap

- Метрики retrieval (Recall@K, MRR)  
- Chat-интерфейс  
- Streaming-ответы LLM
