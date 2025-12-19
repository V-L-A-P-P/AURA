import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc


class QueryExpander:
    """
    Модуль для генерации дополнительных формулировок запроса:
    - Режим 'expand'  — расширение запроса (добавление контекста);
    - Режим 'paraphrase' — перефразирование (изменение формулировки);
    """

    def __init__(
            self,
            model_name: str = "l3lab/L1-Qwen3-8B-Max",
            device: str = "cuda",
            max_new_tokens: int = 75,
            #load_in_8bit: bool = True,
            #load_in_4bit: bool = False
    ):
        print(f"Загружается модель {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            #load_in_8bit=load_in_8bit,
            #load_in_4bit=load_in_4bit
        )

        self.device = device
        self.max_new_tokens = max_new_tokens

    # ---------------------------------------------------------------------

    def _generate(self, prompt: str) -> str:
        """
        Генерация без утечки <think>.
        """

        messages = [
            # IMPORTANT: chain-of-thought is forbidden
            {
                "role": "system",
                "content": (
                    "Respond only with the final result. "
                    "Do not reveal your reasoning process, chain-of-thought, reasoning tokens, "
                    "and do not use <think> tags. "
                    "Answer briefly and without additional comments."
                )
            },
            {"role": "user", "content": prompt}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.05
            )

        text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip().replace("<think>", "").replace("</think>", "")

        # чистим GPU
        torch.cuda.empty_cache()
        gc.collect()

        return text

        text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        return text.strip()

    def _build_expand_prompt(self, query: str, n_variants: int) -> str:
        return f"""
    You must output ONLY the final expanded variants.

CRITICAL OUTPUT CONSTRAINTS (NON-NEGOTIABLE):
- Output MUST contain EXACTLY {n_variants} non-empty lines.
- EACH line MUST be a final expanded sentence in Russian.
- ANY extra word, symbol, or line INVALIDATES the entire output.

ABSOLUTELY FORBIDDEN UNDER ANY CIRCUMSTANCES:
- any reasoning, explanations, analysis, chain-of-thought
- any assumptions, comments, or meta-text
- any mention of what you are doing
- any reference to rules, instructions, or the prompt
- any introductions, lead-ins, transitions, or filler phrases
- any statements about the original query
- any statements about generating or expanding variants
- any summaries, clarifications, or descriptions
- ANYTHING except the final expanded sentences in Russian

EXPLICIT HARD BAN (PRIORITY):
The output MUST NOT begin with, contain, or imply the phrase:
"Okay, let's tackle this."

This ban includes ANY variation, partial form, or semantic equivalent, including but not limited to:
- "Ok, let's tackle this"
- "Okay let's tackle this"
- "Alright, let's tackle this"
- "Let's tackle this"
- "Let's deal with this"
- "Let's work through this"
- "Okay, let's start"
- "Alright, let's begin"

If ANY such phrase or equivalent appears, the ENTIRE OUTPUT is INVALID.

CONVERSATIONAL / DISCOURSE BAN:
The output MUST NOT contain any conversational openings or discourse markers,
including but not limited to:
"Okay", "Alright", "So", "Now", "Let's", "First", "Here", "Below", "Sure".

ROLE (STRICT):
You are a deterministic query expansion engine.
You ONLY expand the original query by adding concrete context.

STRICT SEMANTIC RULES:
1. You MUST NOT change the purpose or intent of the original query.
2. You MUST NOT paraphrase the query — ONLY ADD context.
3. Every variant MUST introduce NEW, concrete information.
4. Each variant MUST add at least one meaningful element:
   action, condition, limitation, example, real usage scenario, or user motivation.
5. You MUST NOT remove or weaken any important information from the original query.

FORM RULES:
- Each variant must be exactly ONE sentence.
- Each variant must be natural and detailed.
- Variants must differ by added context, not by wording only.
- No numbering, bullets, quotes, emojis, or formatting markers.

---

FEW-SHOT EXAMPLES (PATTERN ONLY, NEVER TO BE REFERENCED):

Original query: "Как оплатить кредит?"
Valid output:
Как оплатить кредит через мобильное приложение банка, если временно нет доступа к интернет-банку?
Какие способы доступны для оплаты кредита в выходные или праздничные дни без визита в отделение?
Можно ли оплатить кредит досрочно с карты другого банка без комиссии?

Original query: "Проблемы с входом в личный кабинет"
Valid output:
Почему не получается войти в личный кабинет после недавней смены пароля?
Что делать, если при входе в личный кабинет появляется ошибка «неверный код подтверждения»?
Как восстановить доступ к личному кабинету, если утерян номер телефона, привязанный к входу?

---

Original query: {query}

Generate EXACTLY {n_variants} expanded variants.
Output NOTHING except those final Russian sentences.

    """

    def _build_rephrase_prompt(self, query: str, n_variants: int) -> str:
        return f"""You must output ONLY the final rephrased variants.

CRITICAL OUTPUT CONSTRAINTS (NON-NEGOTIABLE):
- Output MUST consist of EXACTLY {n_variants} non-empty lines.
- EACH line MUST be a valid rephrased variant of the original query.
- ANY extra text INVALIDATES the entire output.

FORBIDDEN CONTENT (ABSOLUTE BAN):
The output MUST NOT contain:
- introductions, lead-ins, acknowledgements, confirmations
- explanations, reasoning, analysis, comments
- meta-text or references to the task, prompt, or model
- formatting markers, quotes, emojis
- numbering, bullets, prefixes, or suffixes
- conversational or discourse phrases of any kind

EXPLICIT HARD BAN (PRIORITY):
The output MUST NOT begin with, contain, or imply the phrase:
"Okay, let's tackle this."

This includes ANY variation, paraphrase, partial form, or semantic equivalent, including but not limited to:
- "Ok, let's tackle this"
- "Okay let's tackle this"
- "Alright, let's tackle this"
- "Let's tackle this"
- "Let's deal with this"
- "Let's work through this"
- "Okay, let's start"
- "Alright, let's begin"

If ANY such phrase or equivalent appears, the ENTIRE OUTPUT is INVALID.

CONVERSATIONAL OPENING BAN:
Any conversational opening or task-introducing construct
(e.g. "Okay", "Alright", "So", "Now", "Let's", "First", "Here are")
appearing ANYWHERE in the output INVALIDATES the response.

ROLE (STRICT):
You are a deterministic query rephrasing engine.
You ONLY transform the user's query into alternative phrasings.

SEMANTIC CONSTRAINTS:
- Preserve the original meaning EXACTLY.
- Do NOT add, remove, generalize, or specialize information.
- Do NOT change intent, scope, assumptions, or domain.
- ALL key terms and core semantic elements MUST appear in EACH variant.

FORM CONSTRAINTS:
- Each variant must be exactly ONE sentence.
- Each variant must be self-contained.
- Variants must differ in wording, syntax, or structure (not punctuation-only).

---

FEW-SHOT EXAMPLES (PATTERN ONLY, NEVER TO BE REFERENCED):

Original query: "Как оплатить кредит?"
Valid output:
Какие способы оплаты кредита доступны?
Как можно внести платеж по кредиту?
Каким образом оплатить кредитный долг?

Original query: "Проблемы с входом в личный кабинет"
Valid output:
Не получается войти в личный кабинет
Ошибка при попытке входа в личный кабинет
Почему не удается авторизоваться в личном кабинете?

---

Original query: {query}

Generate EXACTLY {n_variants} lines.
Output NOTHING except those lines.

        """

    def expand_query(self, query: str, n_variants: int = 3) -> list[str]:
        prompt = self._build_expand_prompt(query, n_variants)
        text = self._generate(prompt)
        variants = [t.strip() for t in text.split("\n") if t.strip()]
        return variants[:n_variants]

    def paraphrase_query(self, query: str, n_variants: int = 3) -> list[str]:
        prompt = self._build_rephrase_prompt(query, n_variants)
        text = self._generate(prompt)
        variants = [t.strip() for t in text.split("\n") if t.strip()]
        return variants[:n_variants]


    def generate(self, query: str, mode: str = "expand", n_variants: int = 3) -> list[str]:
        if mode == "expand":
            return self.expand_query(query, n_variants)
        elif mode == "paraphrase":
            return self.paraphrase_query(query, n_variants)
        else:
            raise ValueError("mode должен быть 'expand' или 'paraphrase'")