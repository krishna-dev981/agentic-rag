"""
state.py — Agentic RAG state schema.
LangGraph lo state ni define chestundi — prathi node update chestundi.
"""

from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Agentic RAG graph state.

    Attributes:
        question: User question (current)
        original_question: User original question (rewrite ki mundu)
        documents: Retrieved document chunks
        generation: LLM answer
        retries: Number of retrieval retries
    """
    question: str
    original_question: str
    documents: List[str]
    generation: str
    retries: int