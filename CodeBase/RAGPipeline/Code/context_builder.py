"""
context_builder.py  —  Formats retrieved docs into LLM-ready context
=====================================================================
Responsibilities:
  1. Deduplicates and orders docs by section number
  2. Applies token budget (truncates if needed)
  3. Formats as structured XML for unambiguous citation
  4. Builds the final prompt string passed to the solver nodes
"""

from langchain_core.documents import Document


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SECTION SORTER
# ─────────────────────────────────────────────────────────────────────────────

def _section_sort_key(doc: Document) -> tuple:
    """Sort by chapter then section number, e.g. ('Chapter 8', (8, 2))"""
    chapter = doc.metadata.get("chapter", "")
    sec_num = doc.metadata.get("section_num", "0.0")
    try:
        parts = tuple(int(x) for x in sec_num.split("."))
    except ValueError:
        parts = (0,)
    return (chapter, parts)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TOKEN BUDGET  (rough: 1 token ≈ 4 chars)
# ─────────────────────────────────────────────────────────────────────────────

def _truncate_to_budget(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[... truncated ...]"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CITATION METADATA BLOCK
# ─────────────────────────────────────────────────────────────────────────────

def _format_citation(doc: Document) -> str:
    m = doc.metadata
    # NOTE: section/section_num intentionally excluded — we don't want the LLM
    # to echo internal NCERT section numbers (e.g. "8.1", "8.3") in its answers.
    return (
        f"book={m.get('book_name','?')} | "
        f"chapter={m.get('chapter','?')} | "
        f"page={m.get('page','?')} | "
        f"topic={m.get('topic','?')}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MAIN FORMATTER
# ─────────────────────────────────────────────────────────────────────────────

def build_context(
    docs:           list[Document],
    max_tokens:     int  = 3000,
    per_doc_tokens: int  = 800,
    xml_wrap:       bool = True,
) -> str:
    """
    Takes a list of retrieved Documents and returns a formatted context string.

    Args:
        docs           : Retrieved parent sections from HierarchicalRetriever.
        max_tokens     : Total budget for the entire context block.
        per_doc_tokens : Per-document token budget before truncation.
        xml_wrap       : Wrap each passage in <passage> tags for structured prompting.

    Returns:
        A formatted string ready to be inserted into the LLM prompt.
    """
    if not docs:
        return "No relevant context found in the knowledge base."

    # Sort: chapter → section order
    sorted_docs = sorted(docs, key=_section_sort_key)

    # Deduplicate by chunk_id / section_id
    seen: set[str] = set()
    unique_docs: list[Document] = []
    for doc in sorted_docs:
        uid = doc.metadata.get("chunk_id") or doc.metadata.get("section_id", id(doc))
        if uid not in seen:
            unique_docs.append(doc)
            seen.add(uid)

    # Format each passage
    passages = []
    total_chars = 0
    total_budget = max_tokens * 4

    for i, doc in enumerate(unique_docs):
        if total_chars >= total_budget:
            break

        text = _truncate_to_budget(doc.page_content, per_doc_tokens)
        citation = _format_citation(doc)

        if xml_wrap:
            passage = (
                f'<passage index="{i+1}">\n'
                f'  <citation>{citation}</citation>\n'
                f'  <content>{text}</content>\n'
                f'</passage>'
            )
        else:
            passage = f"[{i+1}] {citation}\n\n{text}"

        passages.append(passage)
        total_chars += len(passage)

    joined = "\n\n".join(passages)
    return joined


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SYSTEM PROMPT TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are a 9th-grade science and mathematics tutor.

Answer the student's question using ONLY the provided context passages below.
If the context does not contain enough information to answer, say so clearly —
do not make up information.

Rules:
- For numerical problems: show all steps, label each step, include units.
- For concept questions: explain in simple language with a real-life example.
- Do NOT mention section numbers, section names, or any internal book references
  (e.g. do not write "Section 8.1", "8.3 Inertia and Mass", or any similar citation).
- Keep the answer suitable for a 9th-grade student.

<context>
{context}
</context>
"""

def build_prompt(context_docs: list[Document], query: str) -> tuple[str, str]:
    """
    Returns (system_prompt, user_message) ready for the LLM.
    """
    context_str  = build_context(context_docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_str)
    return system_prompt, query


# ─────────────────────────────────────────────────────────────────────────────
# 6.  IS-SUFFICIENT CHECK  (used by class_based_agent)
# ─────────────────────────────────────────────────────────────────────────────

def is_context_sufficient(
    docs:             list[Document],
    scored_leaves:    list[tuple[Document, float]],
    min_docs:         int   = 1,
    min_avg_score:    float = 0.20,
    min_positive:     int   = 1,       # at least N leaves must score above 0
) -> bool:
    """
    Returns True if the retrieved context is good enough to answer the query.
    Used to set AgentState.needs_more_context.

    Args:
        docs           : Expanded parent docs from HierarchicalRetriever.
        scored_leaves  : Leaf docs with relevance scores (already clamped to [0,1]).
        min_docs       : Minimum number of parent docs needed.
        min_avg_score  : Minimum average leaf relevance score.
        min_positive   : Minimum number of leaves with score > 0. Prevents out-of-book
                         queries from passing purely on clamped-zero averages when a few
                         marginally relevant docs inflate the count.
    """
    if len(docs) < min_docs:
        return False
    if not scored_leaves:
        return False
    positive_leaves = [s for _, s in scored_leaves if s > 0.0]
    if len(positive_leaves) < min_positive:
        return False
    avg_score = sum(s for _, s in scored_leaves) / len(scored_leaves)
    return avg_score >= min_avg_score