"""
nodes.py — Agentic RAG nodes.
Prathi node state update chestundi (LangGraph lo).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from typing import Literal

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from state import GraphState


# ---------- Config ----------
BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "vectorstore" / "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"
TOP_K = 4


# ---------- Load Resources (once) ----------
print("📂 Loading FAISS index...")
_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
_vectorstore = FAISS.load_local(
    str(VECTOR_DIR),
    _embeddings,
    allow_dangerous_deserialization=True,
)
_retriever = _vectorstore.as_retriever(search_kwargs={"k": TOP_K})
print("✅ Retriever ready")

print(f"🧠 Loading LLM: {LLM_MODEL}...")
_llm = OllamaLLM(model=LLM_MODEL, num_ctx=2048)   # ← ADD
print("✅ LLM ready\n")


# ---------- Node 1: Route Query ----------
ROUTER_PROMPT = """You are a router. Decide if the user's question needs 
document retrieval or can be answered directly.

If the question is about RAG papers (CoRAG, AstroRAG, ORDER, RAG), 
respond with: retrieve
If the question is a greeting or simple chat, respond with: direct

Question: {question}

Respond with ONLY one word: 'retrieve' or 'direct'."""


def route_query(state: GraphState) -> Literal["retrieve", "direct"]:
    """Decide: retrieve from FAISS or direct answer?"""
    print("🔀 Node: Route Query")
    question = state["question"]

    prompt = ROUTER_PROMPT.format(question=question)
    response = _llm.invoke(prompt).strip().lower()

    if "retrieve" in response:
        print("   → Route: RETRIEVE")
        return "retrieve"
    else:
        print("   → Route: DIRECT")
        return "direct"


# ---------- Node 2: Retrieve ----------
def retrieve(state: GraphState) -> GraphState:
    """FAISS nunchi relevant chunks retrieve cheyyi."""
    print("📚 Node: Retrieve")
    question = state["question"]
    docs = _retriever.invoke(question)
    print(f"   → {len(docs)} chunks retrieved")
    return {**state, "documents": [d.page_content for d in docs]}   


# ---------- Node 3: Grade Documents ----------
GRADER_PROMPT = """You are a grader. Check if the document is relevant 
to the question about RAG research papers.

Question: {question}
Document: {document}

If the document mentions ANY related concept (RAG, retrieval, CoRAG, AstroRAG, 
chunking, embeddings, etc.), respond with: yes
Otherwise respond with: no

Respond with ONLY one word: 'yes' or 'no'."""

def grade_documents(state: GraphState) -> GraphState:
    """Embedding-based grading — top-2 docs accept cheyyi."""
    print("📊 Node: Grade Documents (embedding-based)")
    import numpy as np

    question = state["question"]
    documents = state["documents"]

    # Edge case: docs lekapothe empty return
    if not documents:
        print("   → No documents to grade")
        return {"documents": []}

    query_emb = _embeddings.embed_query(question)

    # Prathi doc ki similarity score calculate cheyyi
    scored_docs = []
    for doc in documents:
        doc_emb = _embeddings.embed_query(doc[:1000])
        dot = np.dot(query_emb, doc_emb)
        norm = np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
        similarity = dot / norm if norm > 0 else 0
        scored_docs.append((similarity, doc))

    # Sort by similarity (highest first)
    scored_docs.sort(reverse=True, key=lambda x: x[0])

    # Top-2 docs accept cheyyi (leda anni, 2 kanna thakkuva unte)
    top_n = min(2, len(scored_docs))
    top_docs = [doc for _, doc in scored_docs[:top_n]]

    # Safe print — scores empty aithe skip
    if len(scored_docs) >= 2:
        print(f"   → Top-2 docs accepted (scores: {scored_docs[0][0]:.3f}, {scored_docs[1][0]:.3f})")
    elif len(scored_docs) == 1:
        print(f"   → Top-1 doc accepted (score: {scored_docs[0][0]:.3f})")
    else:
        print("   → No documents accepted")

    return {"documents": top_docs}


def decide_to_generate(state: GraphState) -> Literal["generate", "rewrite"]:
    """Decision: relevant docs unte generate, lekapothe rewrite (max 2 retries)."""
    print("🔀 Node: Decide to Generate")
    documents = state["documents"]
    retries = state.get("retries", 0)

    if len(documents) > 0:
        print("   → Relevant docs → generate")
        return "generate"
    elif retries >= 2:
        print(f"   → Max retries ({retries}) reached → generate anyway")
        return "generate"
    else:
        print(f"   → No relevant docs (retry {retries}) → rewrite")
        return "rewrite"


# ---------- Node 4: Rewrite Query ----------
REWRITE_PROMPT = """Rewrite the user's question to be more specific 
and better for document retrieval.

Original question: {question}

Rewritten question (one line only):"""


def rewrite_query(state: GraphState) -> GraphState:
    """Question ni improve cheyyi — better retrieval kosam."""
    print("✏️  Node: Rewrite Query")
    question = state["question"]
    retries = state.get("retries", 0)

    prompt = REWRITE_PROMPT.format(question=question)
    new_question = _llm.invoke(prompt).strip()
    print(f"   → Original: {question}")
    print(f"   → Rewritten: {new_question}")

    return {"question": new_question, "retries": retries + 1}


# ---------- Node 5: Generate ----------
GENERATE_PROMPT = """You are a helpful research assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."
Always cite the source document name and page number.

Context:
{context}

Question: {question}

Answer:"""


def generate(state: GraphState) -> GraphState:
    """LLM tho answer generate cheyyi."""
    print("💬 Node: Generate")
    question = state["question"]
    documents = state["documents"]

    context = "\n\n".join(documents)
    prompt = GENERATE_PROMPT.format(context=context, question=question)
    generation = _llm.invoke(prompt)

    print(f"   → Answer generated ({len(generation)} chars)")
    return {"generation": generation}


# ---------- Node 6: Direct Answer ----------
def direct_answer(state: GraphState) -> GraphState:
    """Simple question ki direct answer."""
    print("💬 Node: Direct Answer")
    question = state["question"]
    generation = _llm.invoke(question)
    return {"generation": generation}