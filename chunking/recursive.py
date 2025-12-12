from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


def recursive_chunking(
    text: str,
    chunk_size: int,
    overlap: int,
    separators: List[str] | None = None,
) -> List[str]:
    """
    Recursive character-based chunking using LangChain splitter.
    """

    if not isinstance(text, str) or not text.strip():
        return []

    if separators is None:
        separators = ["\n\n", "\n", ".", "?", "!", ";", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=separators,
        length_function=len,
    )

    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c and c.strip()]
