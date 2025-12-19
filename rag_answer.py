import logging
from retrieval.retriever import Retriever
from llm.answer_generator import AnswerGenerator, AnswerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def answer_query(query: str, top_k: int = 5):
    retriever = Retriever()
    hits = retriever.search(query, top_k=top_k)

    gen = AnswerGenerator(AnswerConfig(
        #model_name="Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens=256,
        temperature=0.2,
        context_char_limit=9000,
    ))

    result = gen.generate(query, hits)

    return {
        "query": query,
        "hits": hits,
        "answer": result["answer"],
    }

if __name__ == "__main__":
    q = "Как оплатить кредит через приложение?"
    res = answer_query(q, top_k=5)
    print("\nANSWER:\n", res["answer"])
