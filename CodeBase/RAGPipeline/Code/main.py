"""
main.py  —  Rootwise LangGraph Educational AI  (v2)
=====================================================
Pipeline redesign:

  CONCEPT queries:
    guardrail → RAG retrieve → personalize context → concept_understander → respond

  NUMERICAL queries:
    guardrail → RAG retrieve (formulas only) → book_numerical_solver
                 ↓ (if out of scope)
              ask_user_confirmation → (yes) general_numerical_solver
                                    → (no)  polite_refusal

  QUESTION GENERATION queries:
    guardrail → RAG retrieve (topic numericals) → personalized_question_generator → respond

  OFF-TOPIC / NON-ACADEMIC:
    guardrail → guardrail_refusal (hard block, no LLM call)

Guardrails:
  - Input: reject non-academic queries before any LLM call
  - Numerical: solver ONLY uses formulas found in retrieved book context
  - Output: strip any content that leaks non-academic material
"""

import os
from typing import TypedDict, Literal, Optional

import time
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from retriever import load_retriever, HierarchicalRetriever, filter_by_chapter
from context_builder import build_context, build_prompt, is_context_sufficient
from ingest import ChapterMeta, ingest

import os
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
groq_key = os.getenv("GROQ_API_KEY")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  SHARED STATE
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Input
    query:              str
    user_id:            str
    grade:              str

    # Classification & routing
    query_type:         Literal["concept", "numerical", "question_gen", "off_topic"]
    is_in_book_scope:   bool        # True if retrieved context covers the query
    user_confirmed_oob: Optional[bool]   # None = not asked yet; True/False = user replied

    # RAG
    rag_context:        Optional[str]
    rag_docs_meta:      Optional[list[dict]]
    needs_more_context: bool

    # Personalisation
    personalized_data:  Optional[dict]

    # Output
    response:           str
    guardrail_blocked:  bool        # True if blocked at input guardrail


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SINGLETONS
# ─────────────────────────────────────────────────────────────────────────────

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key= groq_key,
)

_retriever: Optional[HierarchicalRetriever] = None

def get_retriever() -> HierarchicalRetriever:
    global _retriever
    if _retriever is None:
        _retriever = load_retriever(
            leaf_k           = 6,
            use_mmr          = True,
            mmr_fetch_k      = 20,
            score_threshold  = 0.35,
            expand_to_parent = True,
        )
    return _retriever


def get_personalized_data(user_id: str) -> dict:
    """Stub — replace with real DB / user-profile service."""
    return {
        "user_id":          user_id,
        "weak_topics":      ["Newton's laws", "momentum calculations"],
        "strong_topics":    ["basic algebra", "velocity concepts"],
        "learning_style":   "visual",
        "past_errors":      ["forgetting to include units", "sign errors in F=ma"],
        "interests":        ["swimming", "cycling"],   # used for personalised question flavour
        "preferred_examples": ["sports", "everyday objects"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GUARDRAIL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Topics that are always off-limits regardless of framing
_OFF_TOPIC_SIGNAL = """You are an academic content guardrail for a school-level AI tutor.

Determine if the query is a legitimate academic school-level question (science, maths, 
social science, language, history, geography, general knowledge related to school curriculum).

Reply with ONLY one word:
- "academic"   → question is about school curriculum subjects
- "off_topic"  → question is unrelated to academics (entertainment, personal advice, 
                  politics, harmful content, coding projects, adult topics, etc.)

Query: "{query}"
"""

def _is_academic(query: str) -> bool:
    result = llm.invoke([HumanMessage(content=_OFF_TOPIC_SIGNAL.format(query=query))])
    verdict = result.content.strip().lower()
    return "academic" in verdict


# ─────────────────────────────────────────────────────────────────────────────
# 4.  NODES
# ─────────────────────────────────────────────────────────────────────────────

# ── 4a. Input guardrail ──────────────────────────────────────────────────────

def input_guardrail(state: AgentState) -> AgentState:
    """
    Gate 1: reject non-academic queries before any RAG or solver work.
    Sets guardrail_blocked = True if the query is off-topic.
    """
    if not _is_academic(state["query"]):
        return {
            **state,
            "guardrail_blocked": True,
            "query_type":        "off_topic",
            "response": (
                "I'm Rootwise, your academic tutor. I can only help with "
                "school curriculum topics like science, maths, and social science. "
                "Your question seems to be outside that scope — feel free to ask "
                "anything from your textbooks!"
            ),
        }
    return {**state, "guardrail_blocked": False}


# ── 4b. Intent classifier + RAG retrieval ───────────────────────────────────

def classify_and_retrieve(state: AgentState) -> AgentState:
    """
    1. Classify query into concept / numerical / question_gen
    2. Retrieve relevant book context via HierarchicalRetriever
    3. Determine if the book actually covers this query (in_scope check)
    """
    # ── classify intent ──
    classify_prompt = f"""Classify this academic query into ONE of three types:
- "concept"       : student wants a concept explained (definition, how/why something works)
- "numerical"     : student wants a calculation / problem solved
- "question_gen"  : student is asking you to generate practice questions on a topic

Query: "{state['query']}"
Respond with ONLY one word (concept / numerical / question_gen)."""

    classification = llm.invoke([HumanMessage(content=classify_prompt)])
    raw = classification.content.strip().lower()
    query_type = raw if raw in ("concept", "numerical", "question_gen") else "concept"

    # ── RAG retrieval ──
    retriever    = get_retriever()
    context_docs, leaf_docs = retriever.retrieve(state["query"])
    scored_leaves           = retriever.retrieve_with_scores(state["query"])
    sufficient              = is_context_sufficient(context_docs, scored_leaves)

    rag_context   = build_context(context_docs)
    rag_docs_meta = [
        {
            "section": d.metadata.get("section", ""),
            "page":    d.metadata.get("page", ""),
            "chapter": d.metadata.get("chapter", ""),
            "book":    d.metadata.get("book_name", ""),
        }
        for d in context_docs
    ]

    return {
        **state,
        "query_type":         query_type,
        "rag_context":        rag_context,
        "rag_docs_meta":      rag_docs_meta,
        "needs_more_context": not sufficient,
        "is_in_book_scope":   sufficient,
    }


# ── 4c. Personalization enricher (concept + question_gen path) ──────────────

def personalize_context(state: AgentState) -> AgentState:
    """
    Enriches RAG context with student profile before passing to solver.
    Used for concept explanations and question generation.
    Adds interest-tailored framing cues to rag_context.
    """
    profile = get_personalized_data(state["user_id"])

    interest_hint = (
        f"\n\n[PERSONALISATION HINT — for tutor use only]\n"
        f"Student interests: {profile['interests']}\n"
        f"Preferred example contexts: {profile['preferred_examples']}\n"
        f"Weak areas: {profile['weak_topics']}\n"
        f"Past errors: {profile['past_errors']}\n"
        f"Learning style: {profile['learning_style']}\n"
        f"[END HINT]"
    )

    enriched_context = (state.get("rag_context") or "") + interest_hint

    return {
        **state,
        "rag_context":       enriched_context,
        "personalized_data": profile,
    }


# ── 4d. Concept understander ────────────────────────────────────────────────
def concept_understander(state: AgentState) -> AgentState:
    # Use rag_context if ANY docs were retrieved, regardless of scope flag
    has_context = (
        state.get("rag_docs_meta") and len(state["rag_docs_meta"]) > 0
    )

    system = f"""You are Rootwise, a friendly {state.get('grade', '9th')}-grade science/maths tutor.

STRICT RULES:
1. Answer ONLY from the provided textbook context. Do NOT use outside knowledge.
2. If the context is empty or clearly irrelevant, say:
   "This topic isn't covered in your current book chapters. Please check with your teacher."
3. If the context is relevant even partially, use it to answer as best you can.
4. NEVER mention section numbers or section names from the book (e.g. do NOT write
   "8.1", "Section 8.2", "8.3 Inertia and Mass", or any similar reference). Explain
   concepts in your own words without citing internal book structure.
...
<context>
{state.get('rag_context', 'No context retrieved.') if has_context else 'No context retrieved.'}
</context>"""

    result = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["query"]),
    ])

    # ── output guardrail: append weak-area note ──
    profile = state.get("personalized_data") or get_personalized_data(state["user_id"])
    note = _build_personalized_note(result.content, profile, state.get("rag_docs_meta"))

    return {**state, "response": note, "personalized_data": profile}


# ── 4e. Book numerical solver ────────────────────────────────────────────────

def book_numerical_solver(state: AgentState) -> AgentState:
    """
    Solves numericals using ONLY formulas present in the retrieved book context.
    If no formula is found, marks is_in_book_scope = False for re-routing.
    """
    system = f"""You are Rootwise, a strict {state.get('grade', '9th')}-grade science/maths tutor.

STRICT RULES — READ CAREFULLY:
1. You may ONLY use formulas and values that appear EXPLICITLY in the provided textbook context.
2. DO NOT use formulas from general knowledge or memory.
3. If the required formula is NOT in the context, respond with EXACTLY this phrase and nothing else:
   "OUT_OF_BOOK_SCOPE"
4. If the formula IS in the context:
   - State the formula and cite its section + page.
   - Show every calculation step with units.
   - Box the final answer clearly.
   - Do NOT skip steps.
5. Do NOT solve questions outside school science/maths.
6. NEVER mention section numbers or section names from the book (e.g. do NOT write
   "8.1", "Section 8.4", "8.1 A constant force", or anything similar). Just explain
   and solve without citing internal book structure.

<context>
{state.get('rag_context', 'No context retrieved.')}
</context>"""

    result = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["query"]),
    ])

    response_text = result.content.strip()

    if "OUT_OF_BOOK_SCOPE" in response_text:
        return {
            **state,
            "is_in_book_scope": False,
            "response": response_text,
        }

    # Add personalised note for in-scope answers
    profile = state.get("personalized_data") or get_personalized_data(state["user_id"])
    note    = _build_personalized_note(response_text, profile, state.get("rag_docs_meta"))

    return {
        **state,
        "is_in_book_scope":  True,
        "personalized_data": profile,
        "response":          note,
    }


# ── 4f. Out-of-book confirmation prompt ─────────────────────────────────────

def out_of_book_confirmation(state: AgentState) -> AgentState:
    """
    Tells the student the question is outside the book and asks
    if they still want a general solution.
    Sets user_confirmed_oob based on the PREVIOUS state value
    (caller must inject True/False before re-invoking the graph).
    
    On FIRST pass (user_confirmed_oob is None): returns the confirmation question.
    On SECOND pass (user_confirmed_oob is True/False): passes through for routing.
    """
    if state.get("user_confirmed_oob") is None:
        # First pass — ask the student
        confirmation_msg = (
            "This numerical question uses concepts or formulas that aren't covered "
            f"in your {state.get('grade', '9th')}-grade book chapters I have access to.\n\n"
            "Would you like me to solve it using general knowledge anyway? "
            "Reply **yes** to get a full solution, or **no** to skip it."
        )
        return {**state, "response": confirmation_msg}

    # Second pass — user has replied; routing function will handle branching
    return {**state}


# ── 4g. General numerical solver (out-of-book, user confirmed) ──────────────

def general_numerical_solver(state: AgentState) -> AgentState:
    """
    Solves out-of-book numericals with general knowledge.
    Includes a clear disclaimer that this is beyond the textbook.
    """
    system = f"""You are Rootwise, a {state.get('grade', '9th')}-grade science/maths tutor.

The student confirmed they want a solution even though this isn't in their current book.

RULES:
1. Start with a disclaimer: "Note: This solution goes beyond your current textbook."
2. Solve step-by-step with units at every step.
3. Mention which standard formula/principle you're using.
4. Keep the explanation at {state.get('grade', '9th')}-grade level.
5. Do NOT discuss anything outside school maths/science.
6. Do NOT mention section numbers or section names from the book."""

    result = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["query"]),
    ])

    profile = state.get("personalized_data") or get_personalized_data(state["user_id"])
    note    = _build_personalized_note(result.content, profile, state.get("rag_docs_meta"))

    return {**state, "personalized_data": profile, "response": note}


# ── 4h. Polite refusal (user declined out-of-book) ──────────────────────────

def polite_refusal(state: AgentState) -> AgentState:
    """Student declined out-of-book solution."""
    return {
        **state,
        "response": (
            "No problem! Let's stick to what's in your textbook. "
            "Feel free to ask me any other question from your chapters."
        ),
    }


# ── 4i. Question generator ───────────────────────────────────────────────────

def _extract_question_count(query: str) -> int:
    """Parse the number of questions requested from the query string. Defaults to 3."""
    import re
    match = re.search(r'\b(\d+)\s+question', query, re.IGNORECASE)
    if match:
        return max(1, min(int(match.group(1)), 10))  # clamp between 1 and 10
    return 3


def question_generator(state: AgentState) -> AgentState:
    """
    Generates practice questions on a topic.
    - Pulls ACTUAL numericals from retrieved book context.
    - Flavours them with student interests for motivation.
    - Does NOT invent values or formulas not in the book.
    - Respects the count of questions the student asked for.
    """
    profile = state.get("personalized_data") or get_personalized_data(state["user_id"])
    num_questions = _extract_question_count(state["query"])

    # Build difficulty labels dynamically based on count
    if num_questions == 1:
        difficulty_breakdown = "   - Q1: Direct application (easy)"
    elif num_questions == 2:
        difficulty_breakdown = "   - Q1: Direct application (easy)\n   - Q2: Moderate — one extra step"
    else:
        easy    = max(1, num_questions // 3)
        medium  = max(1, num_questions // 3)
        hard    = num_questions - easy - medium
        parts   = [f"   - Q{i+1}: Direct application (easy)" for i in range(easy)]
        parts  += [f"   - Q{easy+i+1}: Moderate" for i in range(medium)]
        parts  += [f"   - Q{easy+medium+i+1}: Challenge — multi-step" for i in range(hard)]
        difficulty_breakdown = "\n".join(parts)

    system = f"""You are Rootwise, a {state.get('grade', '9th')}-grade science/maths tutor 
creating personalised practice questions.

STRICT RULES:
1. Base ALL questions on the formulas, values, and examples in the provided textbook context.
2. Do NOT invent new physics/chemistry facts outside the context.
3. Personalise question scenarios using the student's interests: {profile['interests']}.
   Example: if the book has F=ma and the student likes cricket, frame the question 
   around a cricket ball being bowled.
4. Generate EXACTLY {num_questions} question(s) with this difficulty spread:
{difficulty_breakdown}
5. For each question: state it clearly, then provide the full worked solution below it.
6. Do NOT cite section numbers or section names (e.g. do NOT write "8.1", "Section 8.3").
7. Do NOT generate questions about topics outside school academics.


<context>
{state.get('rag_context', 'No context retrieved.')}
</context>"""

    result = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["query"]),
    ])

    return {**state, "personalized_data": profile, "response": result.content}


# ── 4j. Guardrail refusal (hard block) ───────────────────────────────────────

def guardrail_refusal(state: AgentState) -> AgentState:
    """Already has response set by input_guardrail — just pass through."""
    return {**state}


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PERSONALISED NOTE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _build_personalized_note(
    raw_answer:   str,
    profile:      dict,
    rag_docs_meta: Optional[list[dict]],
) -> str:
    """
    Appends a short personalised study note to any answer.
    Sends only a tail snippet of the answer to the LLM so it cannot
    accidentally rewrite or summarise it — just appends the note.
    """
    # Only send the last 300 chars as context; enough to anchor the note topic
    answer_tail = raw_answer[-300:].strip()

    system = """You are a personalised learning coach for a school student.
You will receive the TAIL of a completed academic answer (for topic context only).
Your ONLY job is to write a short "📌 Personalised Note" block — do NOT repeat,
rewrite, or summarise any part of the answer. Output ONLY the note itself.

The note must be under 80 words and must:
1. Flag if this topic overlaps with a weak area — be specific.
2. Give ONE study tip matched to the student's learning style.
3. Warn about one past error pattern directly relevant to this topic.
Keep it encouraging and concrete. Start with exactly: 📌 Personalised Note:"""

    user_msg = f"""[Answer tail for topic context — DO NOT repeat this]
...{answer_tail}

Student profile:
- Weak topics : {profile['weak_topics']}
- Strong topics: {profile['strong_topics']}
- Style        : {profile['learning_style']}
- Past errors  : {profile['past_errors']}

Write ONLY the 📌 Personalised Note (no preamble, no repetition of the answer):"""

    for attempt in range(3):
        try:
            result = llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user_msg),
            ])
            note = result.content.strip()
            # Guarantee the separator is clean — never glue onto answer mid-sentence
            return raw_answer.rstrip() + "\n\n" + note
        except Exception as e:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                return raw_answer + "\n\n📌 Personalised Note: (unavailable — rate limit reached)"


# ─────────────────────────────────────────────────────────────────────────────
# 6.  ROUTING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def route_after_guardrail(state: AgentState) -> str:
    if state.get("guardrail_blocked"):
        return "blocked"
    return "proceed"


def route_after_classify(state: AgentState) -> str:
    qt = state.get("query_type", "concept")
    if qt == "numerical":
        return "numerical"
    if qt == "question_gen":
        return "question_gen"
    return "concept"   # default for "concept" and anything unexpected


def route_after_book_solver(state: AgentState) -> str:
    if state.get("is_in_book_scope", True):
        return "done"
    return "confirm_oob"


def route_after_confirmation(state: AgentState) -> str:
    confirmed = state.get("user_confirmed_oob")
    if confirmed is None:
        # First pass — we just asked the question, stop here
        return "waiting"
    if confirmed:
        return "solve_general"
    return "refuse"


# ─────────────────────────────────────────────────────────────────────────────
# 7.  GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("input_guardrail",           input_guardrail)
    graph.add_node("guardrail_refusal",          guardrail_refusal)
    graph.add_node("classify_and_retrieve",      classify_and_retrieve)
    graph.add_node("personalize_context",        personalize_context)
    graph.add_node("concept_understander",       concept_understander)
    graph.add_node("book_numerical_solver",      book_numerical_solver)
    graph.add_node("out_of_book_confirmation",   out_of_book_confirmation)
    graph.add_node("general_numerical_solver",   general_numerical_solver)
    graph.add_node("polite_refusal",             polite_refusal)
    graph.add_node("question_generator",         question_generator)

    # Entry
    graph.set_entry_point("input_guardrail")

    # Guardrail gate
    graph.add_conditional_edges(
        "input_guardrail",
        route_after_guardrail,
        {
            "blocked": "guardrail_refusal",
            "proceed": "classify_and_retrieve",
        },
    )
    graph.add_edge("guardrail_refusal", END)

    # Intent routing — all paths go through personalize_context first
    graph.add_conditional_edges(
        "classify_and_retrieve",
        route_after_classify,
        {
            "concept":      "personalize_context",
            "numerical":    "personalize_context",
            "question_gen": "personalize_context",
        },
    )

    # Personalise context → correct solver/generator based on query_type
    graph.add_conditional_edges(
        "personalize_context",
        lambda s: s.get("query_type", "concept"),
        {
            "concept":      "concept_understander",
            "numerical":    "book_numerical_solver",
            "question_gen": "question_generator",
        },
    )

    graph.add_edge("concept_understander", END)
    graph.add_edge("question_generator",   END)

    # Numerical path
    graph.add_conditional_edges(
        "book_numerical_solver",
        route_after_book_solver,
        {
            "done":        END,
            "confirm_oob": "out_of_book_confirmation",
        },
    )

    # Out-of-book confirmation
    graph.add_conditional_edges(
        "out_of_book_confirmation",
        route_after_confirmation,
        {
            "waiting":      END,          # paused; resume when user replies
            "solve_general":"general_numerical_solver",
            "refuse":       "polite_refusal",
        },
    )

    graph.add_edge("general_numerical_solver", END)
    graph.add_edge("polite_refusal",           END)

    return graph


memory   = MemorySaver()
workflow = build_graph()
app      = workflow.compile(checkpointer=memory)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def run_query(
    query:            str,
    user_id:          str  = "student_001",
    grade:            str  = "9th",
    thread_id:        str  = "default",
    user_confirmed_oob: Optional[bool] = None,
) -> dict:
    """
    Main entry point.

    For out-of-book numerical confirmation flow:
      - First call: run_query("A 5kg object...") → returns confirmation question
      - Second call: run_query("A 5kg object...", user_confirmed_oob=True)  → solves it
                  or run_query("A 5kg object...", user_confirmed_oob=False) → polite refusal
    """
    initial: AgentState = {
        "query":              query,
        "user_id":            user_id,
        "grade":              grade,
        "query_type":         "concept",
        "is_in_book_scope":   True,
        "user_confirmed_oob": user_confirmed_oob,
        "rag_context":        None,
        "rag_docs_meta":      None,
        "needs_more_context": False,
        "personalized_data":  None,
        "response":           "",
        "guardrail_blocked":  False,
    }
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke(initial, config=config)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  ONE-TIME INGESTION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def ingest_book(pdf_path: str, chapter_meta: ChapterMeta = None):
    """Run this ONCE per PDF to build the vector store."""
    if chapter_meta is None:
        chapter_meta = ChapterMeta(
            book_name     = "Science - Class IX",
            chapter       = "Chapter 8",
            chapter_title = "Force and Laws of Motion",
            topic         = "Mechanics",
            start_page    = 1,
            end_page      = 13,
        )
    ingest(pdf_path=pdf_path, chapter_meta=chapter_meta)
    print("Ingestion complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  LOCAL TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # (query, expected_type, user_confirmed_oob)
        ("Explain Newton's second law and also solve: a 3 kg object has 15N force applied. Find acceleration.",   "concept",      None),
        ("Why does a rocket accelerate faster as it burns fuel even if the thrust stays constant?", "concept", None),
        ("A 5 kg object accelerates at 3 m/s². What is the force?", "numerical",   None),
        ("Give me practice 5 questions on laws of motion.",           "question_gen", None),
        ("Who won the IPL 2024?",                                   "off_topic",    None),
        ("What is momentum and how is it calculated?",              "concept",      None),
        ("Is momentum conserved when a gun fires a bullet? Prove it mathematically using values from the book.", "concept", None)
    ]

    for query, expected_type, confirmed in tests:
        print(f"\n{'='*65}")
        print(f"Query  : {query}")
        result = run_query(query, user_id="student_42", thread_id="test_run", user_confirmed_oob=confirmed)
        print(f"Type   : {result['query_type']}  (expected: {expected_type})")
        print(f"Blocked: {result['guardrail_blocked']}")
        print(f"Scope  : {'in-book' if result['is_in_book_scope'] else 'out-of-book'}")
        print(f"Sources: {[m['section'] for m in (result['rag_docs_meta'] or [])]}")
        print(f"\nAnswer:\n{result['response']}")
        time.sleep(5)