# Rootwise — LangGraph Educational RAG Agent

A multi-node agentic RAG pipeline that tutors school students strictly from their own textbook chapters — built with **LangGraph**, a **parent-child hierarchical retriever**, and per-student **personalisation**, all gated by guardrails that keep the tutor on-syllabus and refuse to hallucinate formulas.

This is the LangGraph backend for [Rootwise](https://github.com/Namangarg-01/RootWise), built during my AI Engineer internship at RootVestors.

## Architecture

```mermaid
flowchart TD
    Start([Student query]) --> IG[input_guardrail]

    IG -->|off_topic| GR[guardrail_refusal]
    GR --> End1([Response: out of scope])

    IG -->|academic| CR[classify_and_retrieve<br/>intent classify + RAG retrieve]

    CR --> PC[personalize_context<br/>enrich with student profile]

    PC -->|concept| CU[concept_understander]
    CU --> End2([Response + personalised note])

    PC -->|question_gen| QG[question_generator]
    QG --> End3([N practice questions])

    PC -->|numerical| BNS[book_numerical_solver<br/>formulas from book ONLY]

    BNS -->|in book scope| End4([Response + personalised note])
    BNS -->|OUT_OF_BOOK_SCOPE| OOB[out_of_book_confirmation<br/>ask: solve anyway?]

    OOB -->|waiting for reply| End5([Confirmation question])
    OOB -->|yes| GNS[general_numerical_solver<br/>with disclaimer]
    OOB -->|no| PR[polite_refusal]

    GNS --> End6([Response + personalised note])
    PR --> End7([Polite decline])
```

*(Renders automatically on GitHub. Node names above match the actual function names in [`main.py`](CodeBase/RAGPipeline/Code/main.py) 1:1.)*

## How it works

**1. Input guardrail** — an LLM call classifies the query as `academic` or `off_topic` *before* any retrieval or solving happens. Off-topic queries are hard-blocked with zero further LLM calls.

**2. Classify & retrieve** — the query is classified into `concept`, `numerical`, or `question_gen`, and the hierarchical retriever pulls the most relevant textbook sections.

**3. Personalize context** — every path (concept, numerical, question generation) is enriched with the student's profile (weak topics, learning style, past errors, interests) before it reaches a solver. This context is used to flavour explanations and practice questions, and to append a closing study note.

**4. Type-specific solving:**
- `concept_understander` — explains a concept using *only* the retrieved textbook context.
- `book_numerical_solver` — solves numericals using *only* formulas explicitly present in the retrieved context. If no matching formula exists, it returns `OUT_OF_BOOK_SCOPE` instead of guessing.
- `question_generator` — generates a configurable number of practice questions (parsed from the query, e.g. "give me 5 questions"), pulled from real book numericals and flavoured with the student's interests, with an auto-scaled easy/medium/hard difficulty split.

**5. Out-of-book confirmation loop** — if a numerical falls outside the book's scope, the agent asks the student whether to solve it anyway using general knowledge. The graph pauses at this node (checkpointed via `MemorySaver`) until the student's yes/no answer is passed back in on the next invocation — then routes to `general_numerical_solver` (with an explicit "beyond your textbook" disclaimer) or `polite_refusal`.

**6. Personalised note** — every successful answer ends with a short "Personalised Note" — one sentence flagging overlap with a weak area, one study tip matched to the student's learning style, and a warning about a relevant past error pattern.

## RAG: parent-child hierarchical retrieval

Textbook PDFs are ingested into a 3-level hierarchy:

```
Level 0 — Chapter
Level 1 — Section        (e.g. "8.2 First Law of Motion")   ← parent, returned as context
Level 2 — Semantic chunk (≤512 tokens)                       ← child, what gets embedded
```

Only Level-2 leaf chunks are embedded and searched (Chroma + `all-mpnet-base-v2` sentence embeddings, MMR for diversity). At query time, retrieved leaves are expanded and deduplicated to their Level-1 parent sections, so the LLM gets full, coherent passages instead of fragmented chunks — while search itself stays precise at the leaf level.

An `is_context_sufficient` scope check (minimum doc count + average relevance score) decides whether the book actually covers the query, driving the out-of-book confirmation flow above.

## Tech stack

Python, LangGraph, LangChain, Groq (`llama-3.1-8b-instant`), ChromaDB, HuggingFace sentence-transformers (`all-mpnet-base-v2`), PyMuPDF.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with:

```
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

Ingest a textbook PDF once to build the vector store:

```python
from main import ingest_book
ingest_book("path/to/chapter.pdf")
```

Then query the agent:

```python
from main import run_query

result = run_query("A 5 kg object accelerates at 3 m/s². What is the force?", user_id="student_42")
print(result["response"])

# If the answer comes back OUT_OF_BOOK_SCOPE, resume with the student's choice:
result = run_query("A 5 kg object accelerates at 3 m/s². What is the force?",
                    user_id="student_42", user_confirmed_oob=True)
```

## Project structure

```
CodeBase/RAGPipeline/Code/
├── main.py             # AgentState, all graph nodes, routing, graph assembly
├── retriever.py         # HierarchicalRetriever — parent-child retrieval + MMR
├── ingest.py             # PDF → chapter/section/chunk hierarchy → vector store
└── context_builder.py    # Formats retrieved docs into cited, token-budgeted context
```
