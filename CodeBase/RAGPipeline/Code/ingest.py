"""
ingest.py  —  Hierarchical RAG ingestion pipeline
=================================================
Strategy:
  Level 0  →  Chapter          (parent of everything)
  Level 1  →  Section          (e.g. "8.2 First Law of Motion")
  Level 2  →  Semantic chunk   (≤512 tokens, split inside long sections)

Parent-child retrieval:
  - Only Level-2 chunks are embedded & stored in the vector store
  - Each Level-2 chunk carries a `parent_id` pointing to its Level-1 section
  - At query time, retrieve Level-2 chunks → expand to Level-1 parent for richer context

Metadata schema (per your spec):
  {
    "book_name":    "Science - Class IX",
    "chapter":      "Chapter 8",
    "chapter_title":"Force and Laws of Motion",
    "section":      "8.2 First Law of Motion",
    "section_num":  "8.2",
    "page":         89,
    "topic":        "Mechanics",
    "chunk_level":  2,          # 0=chapter, 1=section, 2=leaf
    "chunk_index":  3,          # position inside the section
    "parent_id":    "ch8_s8.2", # section-level doc id
    "chunk_id":     "ch8_s8.2_c3"
  }
"""

import os

# Force transformers/sentence-transformers to skip their TensorFlow backend —
# this project only uses PyTorch embeddings. Without this, environments that
# happen to have TensorFlow + Keras 3 installed (e.g. a shared anaconda base
# env) fail on import with a Keras-3-incompatibility ValueError, even though
# TF is never actually used here.
os.environ.setdefault("USE_TF", "0")

import re
import uuid
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import fitz                                    # pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # swap for voyage/anthropic if needed
from langchain_chroma import Chroma
from langchain_core.documents import Document

BASE = os.path.dirname(os.path.abspath(__file__))

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'} # Change to 'cuda' if you have a GPU
encode_kwargs = {'normalize_embeddings': True}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChapterMeta:
    book_name:     str
    chapter:       str          # "Chapter 8"
    chapter_title: str          # "Force and Laws of Motion"
    topic:         str          # "Mechanics"
    start_page:    int
    end_page:      int


@dataclass
class SectionChunk:
    """Level-1 parent node."""
    section_id:    str          # "ch8_s8.2"
    section_num:   str          # "8.2"
    section_title: str          # "First Law of Motion"
    text:          str
    page_start:    int
    page_end:      int
    meta:          ChapterMeta


@dataclass
class LeafChunk:
    """Level-2 child node — what gets embedded."""
    chunk_id:    str            # "ch8_s8.2_c3"
    parent_id:   str            # → SectionChunk.section_id
    chunk_index: int
    text:        str
    page:        int
    meta:        ChapterMeta
    section_num: str
    section:     str            # full "8.2 First Law of Motion"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PDF TEXT EXTRACTION  (pymupdf — preserves page numbers)
# ─────────────────────────────────────────────────────────────────────────────

def extract_pages(pdf_path: str) -> list[dict]:
    """
    Returns list of { "page": int, "text": str } dicts.
    Page numbers are 1-indexed to match printed book pages.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        # Clean up common PDF artifacts
        text = re.sub(r'\n{3,}', '\n\n', text)   # collapse excessive blank lines
        text = re.sub(r'Reprint \d{4}-\d{2}\s*', '', text)  # remove reprint stamps
        pages.append({"page": i + 1, "text": text.strip()})
    doc.close()
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SECTION DETECTION  (regex-based, NCERT-style headings)
# ─────────────────────────────────────────────────────────────────────────────

# Matches:  "8.1 Balanced and Unbalanced Forces"
#           "8.4 Second Law of Motion"
SECTION_PATTERN = re.compile(
    r'^(?P<num>\d+\.\d+)\s+(?P<title>[A-Z][^\n]{3,60})$',
    re.MULTILINE
)

# Matches:  "Chapter 8"  /  "8"  followed by chapter title on next line
CHAPTER_PATTERN = re.compile(
    r'(?:Chapter\s*)?(?P<num>\d+)\s*\n\s*(?P<title>[A-Z][A-Z\s]+)',
    re.MULTILINE
)


def detect_sections(pages: list[dict]) -> list[dict]:
    """
    Scans all pages and returns section boundaries:
    [{ "num": "8.1", "title": "Balanced and Unbalanced Forces",
       "page_start": 2, "text_start": <char offset in joined text> }, ...]
    """
    # Join all text with page markers for offset tracking
    joined_parts = []
    page_offsets = {}   # char_offset → page_number
    offset = 0
    for p in pages:
        page_offsets[offset] = p["page"]
        joined_parts.append(p["text"])
        offset += len(p["text"]) + 1   # +1 for \n separator

    full_text = "\n".join(joined_parts)

    sections = []
    for m in SECTION_PATTERN.finditer(full_text):
        # Find which page this offset belongs to
        char_pos = m.start()
        page = max(
            (pg for off, pg in page_offsets.items() if off <= char_pos),
            default=1
        )
        sections.append({
            "num":        m.group("num"),
            "title":      m.group("title").strip(),
            "page_start": page,
            "char_start": m.start(),
            "char_end":   m.end(),
        })

    # Compute text spans between section headers
    for i, sec in enumerate(sections):
        start = sec["char_end"]
        end   = sections[i + 1]["char_start"] if i + 1 < len(sections) else len(full_text)
        sec["text"] = full_text[start:end].strip()

        # Estimate end page
        end_char = end
        sec["page_end"] = max(
            (pg for off, pg in page_offsets.items() if off <= end_char),
            default=sec["page_start"]
        )

    return sections, full_text


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SEMANTIC SPLITTER  (for long sections)
# ─────────────────────────────────────────────────────────────────────────────

def make_splitter(
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> RecursiveCharacterTextSplitter:
    """
    Splits on paragraph boundaries first, then sentences, then words.
    Overlap of 64 tokens ensures context continuity at boundaries.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",    # paragraphs  (highest priority)
            "\n",      # lines
            ". ",      # sentence end
            "? ",
            "! ",
            ", ",
            " ",       # word
            "",        # character   (last resort)
        ],
        keep_separator=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  HIERARCHICAL CHUNK BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_chunks(
    sections:    list[dict],
    chapter_meta: ChapterMeta,
    splitter:    RecursiveCharacterTextSplitter,
    min_section_len: int = 200,   # sections shorter than this → single leaf chunk
) -> tuple[list[SectionChunk], list[LeafChunk]]:
    """
    Returns (section_chunks, leaf_chunks).
    section_chunks  = Level-1 parents (not embedded, used for context expansion)
    leaf_chunks     = Level-2 leaves  (embedded into vector store)
    """
    section_chunks: list[SectionChunk] = []
    leaf_chunks:    list[LeafChunk]    = []

    for sec in sections:
        sec_id = f"ch{chapter_meta.chapter.replace('Chapter ', '')}_s{sec['num']}"
        section_title = f"{sec['num']} {sec['title']}"

        # Build Level-1 parent
        s_chunk = SectionChunk(
            section_id    = sec_id,
            section_num   = sec["num"],
            section_title = sec["title"],
            text          = sec["text"],
            page_start    = sec["page_start"],
            page_end      = sec["page_end"],
            meta          = chapter_meta,
        )
        section_chunks.append(s_chunk)

        # Split section text into Level-2 leaves
        if len(sec["text"]) < min_section_len:
            raw_chunks = [sec["text"]]
        else:
            raw_chunks = splitter.split_text(sec["text"])

        for idx, chunk_text in enumerate(raw_chunks):
            if not chunk_text.strip():
                continue
            chunk_id = f"{sec_id}_c{idx}"
            leaf = LeafChunk(
                chunk_id    = chunk_id,
                parent_id   = sec_id,
                chunk_index = idx,
                text        = chunk_text.strip(),
                page        = sec["page_start"],   # approximation; refine if needed
                meta        = chapter_meta,
                section_num = sec["num"],
                section     = section_title,
            )
            leaf_chunks.append(leaf)

    return section_chunks, leaf_chunks


# ─────────────────────────────────────────────────────────────────────────────
# 6.  VECTORSTORE INGESTION
# ─────────────────────────────────────────────────────────────────────────────

def build_documents(
    leaf_chunks:    list[LeafChunk],
    section_chunks: list[SectionChunk],
) -> tuple[list[Document], dict[str, Document]]:
    """
    Converts leaf chunks → LangChain Documents for embedding.
    Returns (leaf_docs, parent_store).
    parent_store: { section_id → Document } used for context expansion at query time.
    """
    leaf_docs = []
    for lc in leaf_chunks:
        meta = asdict(lc.meta)
        doc = Document(
            page_content = lc.text,
            metadata     = {
                **meta,
                "section":      lc.section,
                "section_num":  lc.section_num,
                "page":         lc.page,
                "chunk_level":  2,
                "chunk_index":  lc.chunk_index,
                "parent_id":    lc.parent_id,
                "chunk_id":     lc.chunk_id,
            }
        )
        leaf_docs.append(doc)

    # Parent store — kept in memory or serialised to disk
    parent_store: dict[str, Document] = {}
    for sc in section_chunks:
        meta = asdict(sc.meta)
        parent_store[sc.section_id] = Document(
            page_content = sc.text,
            metadata     = {
                **meta,
                "section":      f"{sc.section_num} {sc.section_title}",
                "section_num":  sc.section_num,
                "page":         sc.page_start,
                "chunk_level":  1,
                "chunk_id":     sc.section_id,
                "parent_id":    None,
            }
        )

    return leaf_docs, parent_store


def ingest(
    pdf_path:        str,
    chapter_meta:    ChapterMeta,
    persist_dir: str = "./chroma_db",
    parent_store_path: str = "./parent_store.json",
    chunk_size:      int  = 512,
    chunk_overlap:   int  = 64,
) -> tuple[Chroma, dict]:
    """
    Full ingestion pipeline. Returns (vectorstore, parent_store).
    """
    print(f"[1/5] Extracting pages from {pdf_path} ...")
    pages = extract_pages(pdf_path)

    print(f"[2/5] Detecting sections ...")
    sections, _ = detect_sections(pages)
    print(f"      Found {len(sections)} sections: {[s['num'] + ' ' + s['title'] for s in sections]}")

    print(f"[3/5] Building hierarchical chunks ...")
    splitter = make_splitter(chunk_size, chunk_overlap)
    section_chunks, leaf_chunks = build_chunks(sections, chapter_meta, splitter)
    print(f"      {len(section_chunks)} section chunks, {len(leaf_chunks)} leaf chunks")

    print(f"[4/5] Building Documents ...")
    leaf_docs, parent_store = build_documents(leaf_chunks, section_chunks)

    print(f"[5/5] Embedding & persisting to {persist_dir} ...")
    embeddings  = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vectorstore = Chroma.from_documents(
        documents          = leaf_docs,
        embedding          = embeddings,
        persist_directory  = persist_dir,
        collection_name    = "edu_rag",
    )

    # Serialise parent store to disk for retrieval-time context expansion
    serialised = {
        k: {"text": v.page_content, "metadata": v.metadata}
        for k, v in parent_store.items()
    }
    Path(parent_store_path).write_text(json.dumps(serialised, indent=2))
    print(f"      Parent store saved → {parent_store_path}")

    return vectorstore, parent_store


# ─────────────────────────────────────────────────────────────────────────────
# 7.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    meta = ChapterMeta(
        book_name     = "Science - Class IX",
        chapter       = "Chapter 8",
        chapter_title = "Force and Laws of Motion",
        topic         = "Mechanics",
        start_page    = 1,
        end_page      = 13,
    )


    vs, ps = ingest(
        pdf_path          = os.path.join(BASE, "..", "Files", "Force_and_laws_of_Motion.pdf"),
        chapter_meta      = meta,
        persist_dir       = os.path.join(BASE, "chroma_db"),
        parent_store_path = os.path.join(BASE, "parent_store.json"),
    )
