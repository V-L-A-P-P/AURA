import logging
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


@dataclass
class AnswerConfig:
    model_name: str = "Qwen/Qwen2.5-1B-Instruct"
    max_new_tokens: int = 384
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    context_char_limit: int = 9000  # prompt overflow protection


class AnswerGenerator:
    def __init__(self, cfg: AnswerConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading LLM: {cfg.model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            use_fast=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        self.model.eval()


    def _build_prompt(self, query: str, docs: List[Dict[str, Any]]) -> str:
        parts = []
        total_chars = 0

        for i, doc in enumerate(docs, start=1):
            text = (doc.get("text") or "").strip()
            if not text:
                continue

            block = (
                f"[DOC {i}] web_id={doc.get('web_id')} "
                f"chunk_id={doc.get('chunk_id')}\n"
                f"{text}\n"
            )

            if total_chars + len(block) > self.cfg.context_char_limit:
                break

            parts.append(block)
            total_chars += len(block)

        context = "\n---\n".join(parts) if parts else "No relevant documents were retrieved."

        return f"""
You are a helpful assistant answering user questions using ONLY the documents provided.

Rules:
- Use ONLY the information from the documents.
- Do NOT invent facts or add external knowledge.
- If the documents do not contain enough information, say so explicitly.
- Cite facts using [DOC k].
- Prefer concise paragraphs over long numbered lists.
- Finish the answer completely. Do NOT stop mid-sentence or mid-list.

User question:
{query}

Documents:
{context}

Answer:
""".strip()


    @staticmethod
    def _looks_truncated(text: str) -> bool:
        if not text:
            return False
        return (
            text.endswith((
                ":", "-", "•", "–",
            ))
            or text.strip().endswith(tuple(f"{i}." for i in range(1, 15)))
        )


    @torch.inference_mode()
    def generate(self, query: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = self._build_prompt(query, docs)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.temperature > 0,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            repetition_penalty=self.cfg.repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        decoded = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        answer = decoded.split("Answer:", 1)[-1].strip()

        if self._looks_truncated(answer):
            logger.warning("Generated answer appears truncated")

        return {
            "answer": answer,
            "prompt_chars": len(prompt),
            "docs_used": len(docs),
            "truncated": self._looks_truncated(answer),
        }
