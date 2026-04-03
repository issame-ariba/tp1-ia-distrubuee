"""
Prompt Engineering for Multi-Agent Systems — assignment demo.

This script demonstrates tokenizer usage, LangChain chat via Groq's OpenAI-compatible
API (ChatOpenAI + custom base URL), IMDB loading, three prompt styles (zero-shot,
few-shot, chain-of-thought), and micro-F1 evaluation against held-out labels.
"""

from __future__ import annotations

import os
import re
from string import Formatter
from typing import Any, Callable

import pandas as pd
import tiktoken
from datasets import load_dataset
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# IMDB labels: 0 = negative, 1 = positive
LABEL_TO_SENTIMENT = {0: "negative", 1: "positive"}

# Few-shot pool size (from the "examples" split)
FEW_SHOT_N = 3

# Random seed for reproducible splits and example sampling
RANDOM_STATE = 42

# Limit evaluation rows to keep API usage reasonable for a student project.
# Set to None to evaluate the entire gold split (can be slow and costly).
MAX_EVAL_SAMPLES = 50

# Groq chat model (OpenAI-compatible endpoint; see Groq model list in console)
CHAT_MODEL = "openai/gpt-oss-120b"

# Groq OpenAI-compatible API base URL
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# System prompt used for tokenizer demo and as base for classification prompts
SYSTEM_PROMPT = (
    "You are a careful assistant that classifies movie review sentiment. "
    "Follow the user's format instructions exactly."
)


# ---------------------------------------------------------------------------
# Tokenizer (tiktoken)
# ---------------------------------------------------------------------------


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Return token count for the given text using tiktoken."""
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


# ---------------------------------------------------------------------------
# Dataset: IMDB → pandas + sentiment column
# ---------------------------------------------------------------------------


def load_imdb_dataframe() -> pd.DataFrame:
    """Load IMDB, convert to pandas, add string sentiment labels."""
    ds = load_dataset("imdb")
    # Use train split for building examples + gold (we split further below)
    df = ds["train"].to_pandas()
    df["sentiment"] = df["label"].map(LABEL_TO_SENTIMENT)
    return df


def split_examples_and_gold(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into two disjoint sets:
    - examples: pool for few-shot sampling
    - gold_examples: held-out ground truth for evaluation
    """
    examples, gold = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df["sentiment"],
    )
    return examples.reset_index(drop=True), gold.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Few-shot helpers (assignment requirement)
# ---------------------------------------------------------------------------


def create_examples(dataset: pd.DataFrame, n: int) -> list[dict[str, Any]]:
    """
    Build n random labeled examples from the given dataset (DataFrame).

    Each example is a dict with keys 'text' and 'sentiment'.
    """
    n = min(n, len(dataset))
    sample = dataset.sample(n=n, random_state=RANDOM_STATE)
    rows: list[dict[str, Any]] = []
    for _, row in sample.iterrows():
        rows.append({"text": str(row["text"]), "sentiment": str(row["sentiment"])})
    return rows


def create_prompt(
    system_message: str,
    examples: list[dict[str, Any]],
    template: str,
    **extra: str,
) -> str:
    """
    Fill a template string with system context and optional few-shot examples.

    Supported placeholders:
    - {system_message}
    - {examples_block} — formatted few-shot block (empty if no examples)
    - Any extra keyword arguments are forwarded to str.format (e.g. review=...).
    """
    if examples:
        lines = []
        for i, ex in enumerate(examples, start=1):
            lines.append(
                f"Example {i}:\nReview: {ex['text']}\nSentiment: {ex['sentiment']}\n"
            )
        examples_block = "\n".join(lines).strip()
    else:
        examples_block = ""

    # Only pass keys that appear in the template — str.format rejects unused kwargs.
    field_names = {name for _, name, _, _ in Formatter().parse(template) if name}
    kwargs: dict[str, str] = {}
    if "system_message" in field_names:
        kwargs["system_message"] = system_message
    if "examples_block" in field_names:
        kwargs["examples_block"] = examples_block
    for key, value in extra.items():
        if key in field_names:
            kwargs[key] = value
    return template.format(**kwargs)


# ---------------------------------------------------------------------------
# LLM + parsing
# ---------------------------------------------------------------------------


def get_llm() -> ChatOpenAI:
    """
    ChatOpenAI pointed at Groq's OpenAI-compatible endpoint.

    Uses GROQ_API_KEY from the environment (or .env via load_dotenv).
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_groq_api_key_here":
        raise RuntimeError(
            "Set GROQ_API_KEY in your environment or .env file (see README). "
            "Get a key at https://console.groq.com/"
        )
    return ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0,
        api_key=api_key,
        base_url=GROQ_BASE_URL,
    )


def invoke_sentiment(llm: ChatOpenAI, user_content: str) -> str:
    """Send system + user messages and return parsed sentiment."""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    resp = llm.invoke(messages)
    content = getattr(resp, "content", str(resp))
    return parse_sentiment_answer(content)


def parse_sentiment_answer(raw: str) -> str:
    """
    Extract 'positive' or 'negative' from model output.
    The prompts ask for a single final label word.
    """
    text = raw.strip().lower()
    # Prefer explicit final-line patterns
    for line in reversed(text.splitlines()):
        line = line.strip()
        if "positive" in line and "negative" not in line:
            return "positive"
        if "negative" in line and "positive" not in line:
            return "negative"
    # Fallback: last word heuristic
    words = re.findall(r"\b(positive|negative)\b", text)
    if words:
        return words[-1]
    return "negative"


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


def build_zero_shot_user_prompt(review: str) -> str:
    """Zero-shot: instructions only, no examples."""
    return (
        "Classify the sentiment of the following movie review.\n"
        "Respond with exactly one word: positive or negative.\n\n"
        f"Review:\n{review}\n"
    )


def build_few_shot_user_prompt(review: str, examples: list[dict[str, Any]]) -> str:
    """Few-shot: prepend labeled examples from the dataset."""
    return create_prompt(
        SYSTEM_PROMPT,
        examples,
        (
            "{examples_block}\n\n"
            "Now classify this new review.\n"
            "Respond with exactly one word: positive or negative.\n\n"
            "Review:\n{review}\n"
        ),
        review=review,
    )


def build_cot_user_prompt(review: str) -> str:
    """Chain-of-thought: ask for brief reasoning, then final label."""
    return (
        "Classify the sentiment of the movie review below.\n"
        "First, write a short reasoning (2-4 sentences) about tone and content.\n"
        "On the last line, output exactly one word: positive or negative\n\n"
        f"Review:\n{review}\n"
    )


# ---------------------------------------------------------------------------
# Evaluation (micro F1)
# ---------------------------------------------------------------------------


def evaluate_prompt(
    llm: ChatOpenAI,
    gold_df: pd.DataFrame,
    name: str,
    predict_fn: Callable[[ChatOpenAI, str], str],
    *,
    max_samples: int | None = None,
    log: Callable[[str], None] | None = None,
    on_row: Callable[[int, int], None] | None = None,
) -> float:
    """
    Loop over gold examples, compare predictions to ground truth, report micro F1.

    `predict_fn` takes (llm, review_text) and returns 'positive' or 'negative'.

    If max_samples is None, uses module-level MAX_EVAL_SAMPLES (may be None = full gold).
    If max_samples is an int, evaluates at most that many rows from the top of gold_df.
    log: if None, prints; otherwise called with status lines (e.g. Streamlit).
    on_row: optional callback(current_index_1based, total) for progress UIs.
    """
    y_true: list[str] = []
    y_pred: list[str] = []

    if max_samples is None:
        subset = gold_df
        if MAX_EVAL_SAMPLES is not None:
            subset = gold_df.head(MAX_EVAL_SAMPLES)
    else:
        subset = gold_df.head(max_samples)

    total = len(subset)
    sink = log if log is not None else print

    for i, (_, row) in enumerate(subset.iterrows(), start=1):
        true_label = str(row["sentiment"])
        pred_label = predict_fn(llm, str(row["text"]))
        y_true.append(true_label)
        y_pred.append(pred_label)
        if on_row is not None:
            on_row(i, total)

    score = f1_score(y_true, y_pred, average="micro", labels=["positive", "negative"])
    sink(f"{name} — micro F1: {score:.4f}")
    return float(score)


def main() -> None:
    load_dotenv()

    # --- Tokenizer demo ---
    n_tokens = count_tokens(SYSTEM_PROMPT)
    print(f"System prompt token count (tiktoken, cl100k_base): {n_tokens}")

    # --- Simple LLM interaction (assignment example) ---
    llm = get_llm()
    demo_msg = [HumanMessage(content="What is an AI Agent?")]
    demo_resp = llm.invoke(demo_msg)
    demo_text = getattr(demo_resp, "content", str(demo_resp))
    print("\n--- LLM demo (ChatOpenAI → Groq) ---")
    print(f"Q: What is an AI Agent?\nA: {demo_text.strip()}\n")

    # --- Dataset ---
    print("Loading IMDB (this may download data on first run)...")
    df = load_imdb_dataframe()
    examples_df, gold_df = split_examples_and_gold(df)
    few_shot_examples = create_examples(examples_df, FEW_SHOT_N)

    print(f"Examples pool rows: {len(examples_df)}")
    print(f"Gold (evaluation) rows: {len(gold_df)}")
    if MAX_EVAL_SAMPLES is not None:
        print(f"Evaluating on first {MAX_EVAL_SAMPLES} gold rows (see MAX_EVAL_SAMPLES).\n")
    else:
        print("Evaluating on full gold split.\n")

    def predict_zero_shot(llm_: ChatOpenAI, review: str) -> str:
        return invoke_sentiment(llm_, build_zero_shot_user_prompt(review))

    def predict_few_shot(llm_: ChatOpenAI, review: str) -> str:
        return invoke_sentiment(
            llm_, build_few_shot_user_prompt(review, few_shot_examples)
        )

    def predict_cot(llm_: ChatOpenAI, review: str) -> str:
        return invoke_sentiment(llm_, build_cot_user_prompt(review))

    print("--- Evaluation (micro F1 vs gold labels) ---")
    evaluate_prompt(llm, gold_df, "Zero-shot", predict_zero_shot)
    evaluate_prompt(llm, gold_df, "Few-shot", predict_few_shot)
    evaluate_prompt(llm, gold_df, "CoT", predict_cot)


if __name__ == "__main__":
    main()
