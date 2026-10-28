from __future__ import annotations

import re
import sys

import streamlit as st

from answer_question import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    build_context,
    generate_answer,
)
from retrieval import retrieve


MAX_QUESTION_LENGTH = 1000
MAX_HISTORY_MESSAGES = 6
SAMPLE_PROMPTS = (
    "How did revenue change from Q1 to Q2 FY2026?",
    "What was the operating margin in Q3 FY2026?",
    "Summarize the key financial highlights for Q2 FY2026.",
    "How did net profit change across FY2026 quarters?",
)
FOLLOW_UP_PATTERN = re.compile(
    r"\b(it|that|those|them|there|same|previous|before|"
    r"what about|how about|and|also|compare|why)\b",
    re.IGNORECASE,
)
CITATION_PATTERN = re.compile(r"\[(S\d+)\]")
PROMPT_OVERRIDE_PATTERNS = (
    re.compile(
        r"\b(?:ignore|disregard|override|bypass|forget)\b.{0,80}"
        r"\b(?:system|developer|previous|prior|above|all)\b.{0,40}"
        r"\b(?:instructions?|rules?|prompts?|guardrails?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:reveal|show|print|repeat|output)\b.{0,40}"
        r"\b(?:system|developer)\s+(?:prompt|instructions?)\b",
        re.IGNORECASE,
    ),
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

st.set_page_config(
    page_title="Financial Reports Chat",
    page_icon="💬",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f4f9ff;
        color: #17324d;
    }
    [data-testid="stHeader"] {
        background: rgba(244, 249, 255, 0.9);
    }
    [data-testid="stSidebar"] {
        display: none;
    }
    [data-testid="stMain"] {
        background: #f4f9ff;
    }
    [data-testid="stAppViewContainer"] {
        background: #f4f9ff;
    }
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottom"] > div > div {
        background: #f4f9ff;
    }
    [data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #dce9f5;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        box-shadow: 0 3px 14px rgba(37, 91, 145, 0.06);
    }
    [data-testid="stChatInput"] {
        background: #ffffff;
        border: 1px solid #c9dced;
        border-radius: 16px;
    }
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] div[data-baseweb="textarea"] {
        background: #ffffff;
    }
    [data-testid="stChatInput"] textarea {
        color: #17324d;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #71869b;
    }
    [data-testid="stButton"] > button {
        min-height: 3.2rem;
        background: #ffffff !important;
        color: #17324d !important;
        border: 1px solid #c9dced !important;
        border-radius: 12px;
        box-shadow: 0 3px 12px rgba(37, 91, 145, 0.06);
        transition: background-color 120ms ease, border-color 120ms ease,
            transform 120ms ease;
    }
    [data-testid="stButton"] > button p {
        color: #17324d !important;
    }
    [data-testid="stButton"] > button:hover {
        background: #e8f4ff !important;
        border-color: #79b9ed !important;
        color: #0c5d9f !important;
        transform: translateY(-1px);
    }
    [data-testid="stButton"] > button:hover p {
        color: #0c5d9f !important;
    }
    [data-testid="stButton"] > button:focus-visible {
        outline: 3px solid #79b9ed;
        outline-offset: 2px;
    }
    h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] {
        color: #17324d;
    }
    a {
        color: #0878d1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def build_retrieval_query(
    question: str,
    history: list[dict[str, str]],
) -> str:
    """Add the last user question for short follow-ups needing context."""
    if not history or not FOLLOW_UP_PATTERN.search(question):
        return question

    previous_question = next(
        (
            message["content"]
            for message in reversed(history)
            if message.get("role") == "user"
        ),
        None,
    )
    if not previous_question:
        return question

    return f"{previous_question}\nFollow-up question: {question}"


def find_unknown_citations(answer: str, citations: list[str]) -> set[str]:
    """Return citation labels used by the model but absent from its evidence."""
    allowed = {
        match.group(1)
        for citation in citations
        if (match := CITATION_PATTERN.search(citation))
    }
    used = set(CITATION_PATTERN.findall(answer))
    return used - allowed


def has_valid_citation(answer: str, citations: list[str]) -> bool:
    allowed = {
        match.group(1)
        for citation in citations
        if (match := CITATION_PATTERN.search(citation))
    }
    return bool(set(CITATION_PATTERN.findall(answer)) & allowed)


def apply_citation_guardrail(
    answer: str,
    citations: list[str],
) -> tuple[str, str | None]:
    """Withhold answers that cite labels absent from retrieved evidence."""
    unknown_citations = find_unknown_citations(answer, citations)

    if unknown_citations:
        return (
            "I couldn't verify the generated answer against the retrieved "
            "report sources, so I won't present it as a supported answer. "
            "Please try rephrasing the question or review the sources.",
            "The answer referenced source labels that were not in the "
            "retrieved evidence, so the generated answer was withheld.",
        )

    if not has_valid_citation(answer, citations):
        return answer, None

    return answer, None


def validate_question(question: str) -> str | None:
    """Return a user-facing rejection reason when a question violates limits."""
    if not question.strip():
        return "Please enter a question."
    if len(question) > MAX_QUESTION_LENGTH:
        return (
            f"Questions must be {MAX_QUESTION_LENGTH} characters or fewer."
        )
    if any(pattern.search(question) for pattern in PROMPT_OVERRIDE_PATTERNS):
        return (
            "I can only answer questions about the indexed financial reports. "
            "Requests to override instructions or reveal hidden prompts are "
            "not supported."
        )
    return None


def initialize_chat() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        sources = message.get("sources", [])
        if sources:
            with st.expander("Report sources"):
                for source in sources:
                    st.markdown(f"- {source}")
        if message.get("citation_warning"):
            st.warning(message["citation_warning"])


def main() -> None:
    initialize_chat()
    st.title("Financial Reports")
    st.caption("Ask a question about Infosys financial reports.")

    selected_prompt = None
    st.markdown("#### Try asking")
    prompt_columns = st.columns(2)
    for index, prompt in enumerate(SAMPLE_PROMPTS):
        with prompt_columns[index % len(prompt_columns)]:
            if st.button(
                prompt,
                key=f"sample_prompt_{index}",
                use_container_width=True,
            ):
                selected_prompt = prompt

    for message in st.session_state.messages:
        render_message(message)

    typed_question = st.chat_input(
        "Ask about revenue, profit, margins, or another report metric...",
        max_chars=MAX_QUESTION_LENGTH,
    )
    question = selected_prompt or typed_question
    if question is None:
        return

    question = question.strip()
    rejection = validate_question(question)
    if rejection:
        st.warning(rejection)
        return

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })
    history = st.session_state.messages[:-1][-MAX_HISTORY_MESSAGES:]
    history_for_model = [
        {
            "role": message["role"],
            "content": message["content"][:MAX_QUESTION_LENGTH],
        }
        for message in history
        if message.get("role") in {"user", "assistant"}
    ]

    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.spinner("Searching reports and preparing an answer..."):
            retrieval_query = build_retrieval_query(question, history)
            results = retrieve(retrieval_query, rerank=True)
            context, citations = build_context(results[:5])
            answer = generate_answer(
                question,
                context,
                model=DEFAULT_MODEL,
                ollama_url=DEFAULT_OLLAMA_URL,
                conversation_history=history_for_model,
            )

        answer, citation_warning = apply_citation_guardrail(answer, citations)
        assistant_message = {
            "role": "assistant",
            "content": answer,
            "sources": citations,
            "citation_warning": citation_warning,
        }
        st.session_state.messages.append(assistant_message)
        render_message(assistant_message)
    except (OSError, RuntimeError, ValueError) as exc:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"I couldn't complete that request: {exc}",
        })
        st.error(f"Couldn't answer the question: {exc}")


if __name__ == "__main__":
    main()
