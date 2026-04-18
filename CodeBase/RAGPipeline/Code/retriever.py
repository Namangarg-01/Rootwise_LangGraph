"""
retriever.py  —  Parent-child retrieval with context expansion
==============================================================
Query flow:
  1. Embed the query
  2. Search leaf chunks (Level-2) by cosine similarity
  3. For each retrieved leaf, fetch its Level-1 parent section
  4. Deduplicate parents (multiple leaves can share one parent)
  5. Return parent sections as the actual context → richer, coherent text

Reranking:
  - MMR (Maximal Marginal Relevance) is used at leaf retrieval to
    reduce redundancy before parent expansion
  - Configurable: swap MMR for a cross-encoder reranker if needed

Metadata filtering:
  - Filter by chapter, section, topic, or page range at query time
"""

import json
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'} # Change to 'cuda' if you have a GPU
encode_kwargs = {'normalize_embeddings': True}


import os
base_path = os.path.dirname(__file__)
parent_store_path = os.path.join(base_path, "parent_store.json")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PARENT STORE LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_parent_store(path):
    p = Path(path)
    if not p.exists():
        print(f"⚠️ Warning: {path} not found. Initializing empty store.")
        return {}
    raw = json.loads(p.read_text())
    return {
        k: Document(page_content=v["text"], metadata=v["metadata"])
        for k, v in raw.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  RETRIEVER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalRetriever:
    """
    Parent-child RAG retriever.

    Args:
        vectorstore:      Chroma vectorstore containing Level-2 leaf chunks.
        parent_store:     Dict mapping section_id → Level-1 Document.
        leaf_k:           Number of leaf chunks to retrieve initially.
        use_mmr:          Use MMR diversity-aware retrieval (True) vs similarity (False).
        mmr_fetch_k:      Pool size for MMR candidate selection.
        score_threshold:  Minimum relevance score (0–1). Leaves below this are dropped.
        expand_to_parent: If True, return parent sections; if False, return leaf chunks.
    """

    def __init__(
        self,
        vectorstore:      Chroma,
        parent_store:     dict[str, Document],
        leaf_k:           int   = 6,
        use_mmr:          bool  = True,
        mmr_fetch_k:      int   = 20,
        score_threshold:  float = 0.35,
        expand_to_parent: bool  = True,
    ):
        self.vs               = vectorstore
        self.parent_store     = parent_store
        self.leaf_k           = leaf_k
        self.use_mmr          = use_mmr
        self.mmr_fetch_k      = mmr_fetch_k
        self.score_threshold  = score_threshold
        self.expand_to_parent = expand_to_parent

    # ── public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:           str,
        filter_meta:     Optional[dict] = None,
    ) -> tuple[list[Document], list[Document]]:
        """
        Returns (context_docs, leaf_docs).
        context_docs: what you pass to the LLM (parents or leaves)
        leaf_docs:    always the raw leaf matches (useful for citations)
        """
        leaf_docs = self._retrieve_leaves(query, filter_meta)

        if not leaf_docs:
            return [], []

        if self.expand_to_parent:
            context_docs = self._expand_to_parents(leaf_docs)
        else:
            context_docs = leaf_docs

        return context_docs, leaf_docs

    def retrieve_with_scores(
        self,
        query:       str,
        filter_meta: Optional[dict] = None,
    ) -> list[tuple[Document, float]]:
        """Returns leaf chunks with their similarity scores (for debugging / reranking)."""
        return self._retrieve_leaves_with_scores(query, filter_meta)

    # ── internals ─────────────────────────────────────────────────────────────

    def _retrieve_leaves(
        self,
        query:       str,
        filter_meta: Optional[dict],
    ) -> list[Document]:
        if self.use_mmr:
            docs = self.vs.max_marginal_relevance_search(
                query,
                k        = self.leaf_k,
                fetch_k  = self.mmr_fetch_k,
                filter   = filter_meta,
            )
        else:
            docs = self.vs.similarity_search(
                query,
                k      = self.leaf_k,
                filter = filter_meta,
            )
        return docs

    def _retrieve_leaves_with_scores(
        self,
        query:       str,
        filter_meta: Optional[dict],
    ) -> list[tuple[Document, float]]:
        import warnings
        # Chroma emits a UserWarning when cosine-distance conversion yields scores
        # outside [0, 1] for low-relevance queries. We clamp them ourselves below,
        # so we suppress the warning here to keep logs clean.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Relevance scores must be between 0 and 1",
                category=UserWarning,
            )
            results = self.vs.similarity_search_with_relevance_scores(
                query,
                k      = self.leaf_k,
                filter = filter_meta,
            )
        # Clamp to [0, 1] — raw cosine distance conversion can produce negative values
        # for clearly off-topic queries; clamping keeps downstream score logic correct.
        clamped = [(doc, max(0.0, min(1.0, score))) for doc, score in results]
        # Apply score threshold
        return [(doc, score) for doc, score in clamped if score >= self.score_threshold]

    def _expand_to_parents(self, leaf_docs: list[Document]) -> list[Document]:
        """
        For each leaf, look up its parent section.
        Deduplicates: multiple leaves from the same section → one parent returned.
        Preserves order of first appearance.
        """
        seen_parents: set[str] = set()
        parents:      list[Document] = []

        for leaf in leaf_docs:
            parent_id = leaf.metadata.get("parent_id")
            if parent_id and parent_id not in seen_parents:
                parent_doc = self.parent_store.get(parent_id)
                if parent_doc:
                    parents.append(parent_doc)
                    seen_parents.add(parent_id)
                else:
                    # Parent not found → fall back to leaf itself
                    parents.append(leaf)
                    seen_parents.add(parent_id or leaf.metadata["chunk_id"])

        return parents


# ─────────────────────────────────────────────────────────────────────────────
# 3.  METADATA FILTER BUILDER  (convenience helpers)
# ─────────────────────────────────────────────────────────────────────────────

def filter_by_chapter(chapter: str) -> dict:
    """e.g. filter_by_chapter("Chapter 8")"""
    return {"chapter": {"$eq": chapter}}


def filter_by_section(section_num: str) -> dict:
    """e.g. filter_by_section("8.2")"""
    return {"section_num": {"$eq": section_num}}


def filter_by_topic(topic: str) -> dict:
    """e.g. filter_by_topic("Mechanics")"""
    return {"topic": {"$eq": topic}}


def filter_by_page_range(start: int, end: int) -> dict:
    return {"page": {"$gte": start, "$lte": end}}


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FACTORY  (loads everything from disk)
# ─────────────────────────────────────────────────────────────────────────────


def load_retriever(
    persist_dir:       str = None,
    parent_store_path: str = None,
    **kwargs,
) -> HierarchicalRetriever:
    base = os.path.dirname(os.path.abspath(__file__))

    if persist_dir is None:
        persist_dir = os.path.join(base, "chroma_db")
    if parent_store_path is None:
        parent_store_path = os.path.join(base, "parent_store.json")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vectorstore = Chroma(
        persist_directory  = persist_dir,
        embedding_function = embeddings,
        collection_name    = "edu_rag",
    )
    parent_store = load_parent_store(parent_store_path)
    return HierarchicalRetriever(vectorstore, parent_store, **kwargs)
# ─────────────────────────────────────────────────────────────────────────────
# 5.  QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    retriever = load_retriever()

    queries = [
        "What is Newton's first law of motion?",
        "How do you calculate force using mass and acceleration?",
        "What is inertia and how does mass relate to it?",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        context_docs, leaf_docs = retriever.retrieve(q)
        print(f"Retrieved {len(leaf_docs)} leaf chunks → expanded to {len(context_docs)} parent sections")
        for i, doc in enumerate(context_docs):
            m = doc.metadata
            print(f"\n  [{i+1}] Section : {m.get('section', 'N/A')}")
            print(f"       Page    : {m.get('page', 'N/A')}")
            print(f"       Chapter : {m.get('chapter', 'N/A')}")
            print(f"       Preview : {doc.page_content[:120].strip()}...")