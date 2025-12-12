import pymorphy3

morph = pymorphy3.MorphAnalyzer()


def lemmatize_text(text: str) -> str:
    return " ".join(morph.parse(w)[0].normal_form for w in text.split())


def clean_text(text: str, lemmatize: bool = True) -> str:
    if not isinstance(text, str):
        return ""

    text = text.replace("\xa0", " ").lower()
    return lemmatize_text(text) if lemmatize else text.strip()
