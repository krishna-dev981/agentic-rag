"""
graph.py — LangGraph workflow for Agentic RAG.
Nodes + edges connect chesi, agentic workflow build chestundi.
"""

import sys
from pathlib import Path

# src/ folder ni path lo add cheyyi (local imports kosam)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langgraph.graph import StateGraph, END

from state import GraphState
from nodes import (
    route_query,
    retrieve,
    grade_documents,
    decide_to_generate,     # NEW
    rewrite_query,
    generate,
    direct_answer,
)


def build_graph():
    """LangGraph workflow build cheyyi."""
    print("🔨 Building LangGraph workflow...\n")

    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)
    workflow.add_node("direct_answer", direct_answer)

    # Entry point — conditional routing
    workflow.set_conditional_entry_point(
        route_query,
        {
            "retrieve": "retrieve",
            "direct": "direct_answer",
        },
    )

    # Edges
    workflow.add_edge("retrieve", "grade_documents")

    # Conditional edge — decide to generate or rewrite
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,     # NEW — separate decision function
        {
            "generate": "generate",
            "rewrite": "rewrite_query",
        },
    )

    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", END)
    workflow.add_edge("direct_answer", END)

    app = workflow.compile()
    print("✅ Graph compiled\n")
    return app


def run_query(app, question: str):
    """Oka question run cheyyi."""
    print("=" * 70)
    print(f"❓ Question: {question}")
    print("=" * 70)

    initial_state = {
        "question": question,
        "original_question": question,
        "documents": [],
        "generation": "",
        "retries": 0,
    }

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 70)
    print("💬 Final Answer:")
    print("=" * 70)
    print(final_state["generation"])
    print("=" * 70)
    print()


def main():
    app = build_graph()

    # Test questions
    questions = [
        "What is CoRAG and how does it work?",
        "Hi, how are you?",
        "Explain the AstroRAG approach.",
    ]

    for q in questions:
        run_query(app, q)


if __name__ == "__main__":
    main()