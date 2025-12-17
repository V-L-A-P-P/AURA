# AURA — Augmented Understanding & Retrieval for Answers

**AURA** — production-oriented RAG-система для поиска и генерации ответов по финтех-корпусу документов.  
Проект реализует **полный end-to-end пайплайн**: от подготовки данных и построения индексов до генерации финального ответа LLM с указанием источников.

Цель проекта — показать, как можно построить **качественный retrieval**, существенно превосходящий базовый RAG, за счёт гибридного поиска, graph-based расширения кандидатов и transformer-based reranking.

---

## 🔹 Ключевые возможности

### Hybrid Retrieval
- TF-IDF (lexical search)
- Dense retrieval через FAISS
- Двухэтапный transformer-based cross-encoder reranking

### Graph-Augmented Retrieval
- Граф связей между чанками документов
- Расширение кандидатов через граф
- Улучшение recall и устойчивости retrieval

### Гибкая подготовка данных
- Несколько стратегий чанкинга:
  - базовый (по словам)
  - семантический
  - рекурсивный
- Препроцессинг документов перед чанкингом

### LLM-driven Query Rewriting
- Переформулировка пользовательских запросов
- Повышение полноты и релевантности поиска

### LLM Answer Generation
- Генерация ответа строго на основе найденных документов
- Контроль контекста и защита от hallucinations
- Явное указание источников (чанков)

### Production-ready архитектура
- Чёткое разделение offline / online этапов
- Тестируемые компоненты (PyTest)
- Подготовка к API и деплою

---

## 🧠 Архитектура пайплайна

```text
Raw documents
    ↓
Preprocessing + Chunking
    ↓
Embeddings + FAISS index
    ↓
Chunk Graph Construction
    ↓
Hybrid Retrieval (TF-IDF + Dense)
    ↓
Graph-based Expansion
    ↓
Transformer Reranking
    ↓
Top-K Documents
    ↓
LLM Answer Generation
```

---

## 📂 Структура проекта

```text
qa-rag-system/
├── run_pipeline.py
├── rag_answer.py
├── api/
│   └── main.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── chunks/
├── embeddings/
│   ├── kb_index.faiss
│   ├── kb_vectors.npy
│   └── kb_metadata.json
├── graph/
│   ├── build_graph.py
│   ├── graph_store.py
│   └── graph_expander.py
├── retrieval/
│   └── retriever.py
├── llm/
│   ├── answer_generator.py
│   └── query_preprocessor.py
├── utils/
│   └── config.py
└── tests/
```

---

## 🚀 Запуск пайплайна

### Полный offline-пайплайн

```bash
python run_pipeline.py
```

---

## 🤖 RAG inference

```python
from rag_answer import answer_query

result = answer_query("Как оплатить кредит через приложение?", top_k=5)
print(result["answer"])
```

---

## 🧪 Тестирование

```bash
pytest
```

---

## 🛠 Технологии

Python, FAISS, Sentence-Transformers, Cross-Encoder, LLM, NetworkX, FastAPI, PyTest

---

## 📌 Roadmap

- Docker
- Retrieval metrics
- UI
- Streaming LLM
