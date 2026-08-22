"""
Rootwise — Streamlit demo UI
============================
A self-contained chat frontend for the LangGraph agent in main.py.
Imports the graph directly (no separate backend needed), so this one
file is enough to demo or deploy on Streamlit Community Cloud.

Run locally:
    streamlit run streamlit_app.py

Deploy on Streamlit Cloud:
    Set GROQ_API_KEY and HF_TOKEN as app secrets (Settings -> Secrets).
"""
import os
import uuid

import streamlit as st

# Streamlit Cloud secrets -> environment, so main.py's os.getenv() picks them up.
if hasattr(st, "secrets"):
    for key in ("GROQ_API_KEY", "HF_TOKEN"):
        try:
            if key in st.secrets and not os.getenv(key):
                os.environ[key] = st.secrets[key]
        except Exception:
            pass

import main as pipeline

st.set_page_config(page_title="Rootwise — AI Tutor", page_icon="📚", layout="centered")

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str, "meta": dict|None}
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None  # query text awaiting yes/no out-of-book confirmation

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Rootwise")
    st.caption("Guardrailed, syllabus-scoped AI tutor — LangGraph + hierarchical RAG")

    st.subheader("Student profile")
    user_id = st.text_input("Student ID", value="student_42")
    grade = st.selectbox("Grade", ["8th", "9th", "10th", "11th", "12th"], index=1)

    st.divider()
    st.caption("Covered chapters (Class IX Science)")
    st.markdown("- Chapter 7 — Motion\n- Chapter 8 — Force and Laws of Motion\n- Chapter 9 — Gravitation")

    st.divider()
    if st.button("🔄 New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.pending_query = None
        st.rerun()

    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is not set. Add it to a .env file locally, or to Streamlit secrets when deployed.")

st.title("Ask Rootwise")
st.caption("Try a concept question, a numerical problem, or ask for practice questions — Rootwise only answers from the chapters above.")

# ── Render existing chat history ────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("meta")
        if meta:
            badges = f"`{meta.get('query_type', '')}`"
            if meta.get("is_in_book_scope") is not None:
                badges += "  ·  " + ("📗 in-syllabus" if meta["is_in_book_scope"] else "📙 outside syllabus")
            st.caption(badges)
            sources = meta.get("sources") or []
            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for s in sources:
                        st.markdown(f"- **{s.get('chapter', '')}** — {s.get('section', '') or 'section n/a'}")


def _sources_from(result) -> list[dict]:
    return result.get("rag_docs_meta") or []


def _history_for_context() -> list[dict]:
    """Prior turns as plain {"role", "content"} dicts for main.py's chat_history param."""
    return [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]


def _run(query: str, confirmed=None, history=None):
    with st.spinner("Thinking..."):
        try:
            return pipeline.run_query(
                query=query,
                user_id=user_id,
                grade=grade,
                thread_id=st.session_state.thread_id,
                user_confirmed_oob=confirmed,
                chat_history=history if history is not None else _history_for_context(),
            )
        except Exception:
            # run_query() already degrades gracefully for known failure modes
            # (LLM down, retriever error). This is a last-resort safety net —
            # a Streamlit crash mid-conversation loses the whole chat history.
            return {
                "query_type": "concept",
                "is_in_book_scope": False,
                "user_confirmed_oob": confirmed,
                "response": "Something went wrong on my end. Please try asking again.",
                "rag_docs_meta": [],
            }


def _is_pending_confirmation(result) -> bool:
    return (
        result.get("query_type") == "numerical"
        and result.get("is_in_book_scope") is False
        and result.get("user_confirmed_oob") is None
    )


# ── Pending out-of-book confirmation buttons ────────────────────────────────
if st.session_state.pending_query:
    col1, col2 = st.columns(2)
    if col1.button("✅ Yes, solve it anyway"):
        history = _history_for_context()  # snapshot BEFORE appending this click
        st.session_state.messages.append({"role": "user", "content": "Yes, solve it anyway.", "meta": None})
        result = _run(st.session_state.pending_query, confirmed=True, history=history)
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.get("response", ""),
            "meta": {"query_type": result.get("query_type"), "is_in_book_scope": result.get("is_in_book_scope"),
                     "sources": _sources_from(result)},
        })
        st.session_state.pending_query = None
        st.rerun()
    if col2.button("❌ No, skip it"):
        history = _history_for_context()
        st.session_state.messages.append({"role": "user", "content": "No, skip it.", "meta": None})
        result = _run(st.session_state.pending_query, confirmed=False, history=history)
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.get("response", ""),
            "meta": {"query_type": result.get("query_type"), "is_in_book_scope": result.get("is_in_book_scope"),
                     "sources": _sources_from(result)},
        })
        st.session_state.pending_query = None
        st.rerun()

# ── Chat input ───────────────────────────────────────────────────────────────
if query := st.chat_input("Ask a question from Motion, Force and Laws of Motion, or Gravitation..."):
    history = _history_for_context()  # snapshot BEFORE appending this query
    st.session_state.messages.append({"role": "user", "content": query, "meta": None})
    with st.chat_message("user"):
        st.markdown(query)

    result = _run(query, history=history)

    with st.chat_message("assistant"):
        st.markdown(result.get("response", ""))

    st.session_state.messages.append({
        "role": "assistant",
        "content": result.get("response", ""),
        "meta": {"query_type": result.get("query_type"), "is_in_book_scope": result.get("is_in_book_scope"),
                 "sources": _sources_from(result)},
    })

    if _is_pending_confirmation(result):
        st.session_state.pending_query = query

    st.rerun()
