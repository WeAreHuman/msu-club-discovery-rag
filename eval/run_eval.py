"""
RAGAS evaluation for the MSU Club Discovery RAG pipeline.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --label after_chunking_change
    python eval/run_eval.py --dataset my_dataset.json --top-k 8

Prerequisites:
    pip install -r eval/requirements-eval.txt
    GROQ_API_KEY set in .env  (RAGAS uses Groq llama-3.3-70b-versatile as evaluator LLM)
    All normal app env vars (PINECONE_API_KEY, GROQ_API_KEY, etc.) also set in .env
    HuggingFace embeddings (all-MiniLM-L6-v2) download ~90 MB on first run, then cached.

Results are saved to eval/results/<timestamp>[_label].json
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import os
import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._answer_correctness import AnswerCorrectness
from ragas.metrics._answer_similarity import AnswerSimilarity
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings as LCHFEmbeddings
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper

from src.rag_engine import RAGEngine
import config

EVAL_LLM_MODEL = "llama-3.3-70b-versatile"
EVAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_eval_dataset(rag: RAGEngine, test_cases: list, top_k: int) -> Dataset:
    """Run each test question through the RAG pipeline and collect inputs for RAGAS."""
    questions, answers, contexts, ground_truths = [], [], [], []

    for i, case in enumerate(test_cases, 1):
        q = case["question"]
        print(f"  [{i}/{len(test_cases)}] {q[:75]}")

        response = rag.query(
            question=q,
            top_k=top_k,
            apply_filters=True,
            return_citations=True,
        )

        questions.append(q)
        answers.append(response["answer"])
        # RAGAS expects contexts as a list of strings (one string per retrieved chunk)
        contexts.append([chunk["text"] for chunk in response["retrieved_chunks"]])
        ground_truths.append(case["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def print_summary(summary: dict, label: str = ""):
    title = f"RAGAS RESULTS{f'  ({label})' if label else ''}"
    width = 58
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
    for metric, score in summary.items():
        bar = "█" * int(score * 30)
        print(f"  {metric:<26} {score:.3f}  {bar}")
    print("=" * width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="dataset.json",
        help="Test dataset filename (relative to eval/). Default: dataset.json",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Short label for this run, e.g. 'baseline' or 'after_prompt_change'",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per question. Default: 5",
    )
    args = parser.parse_args()

    # Guard: GROQ_API_KEY is required — RAGAS uses Groq as its evaluator LLM
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY is not set.")
        print("RAGAS uses Groq (llama-3.3-70b-versatile) to score faithfulness, relevancy, etc.")
        print("Add GROQ_API_KEY to your .env file and retry.")
        sys.exit(1)

    # Load test cases
    dataset_path = Path(__file__).parent / args.dataset
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        sys.exit(1)

    with open(dataset_path) as f:
        test_cases = json.load(f)

    print(f"Loaded {len(test_cases)} test cases from {dataset_path.name}")
    print(f"top_k={args.top_k}  label='{args.label or 'none'}'")
    print(f"Evaluator LLM  : Groq / {EVAL_LLM_MODEL}")
    print(f"Evaluator embed: HuggingFace / {EVAL_EMBED_MODEL}")

    # Init RAG engine (uses Groq + Pinecone from .env)
    config.validate_config()
    rag = RAGEngine()

    # Step 1: collect questions, answers, contexts, ground_truths
    print("\nRunning questions through RAG pipeline...")
    eval_dataset = build_eval_dataset(rag, test_cases, top_k=args.top_k)

    # Step 2: configure RAGAS to use Groq as evaluator LLM + local HuggingFace embeddings
    # Use OpenAI client pointed at Groq's base URL — Groq is OpenAI-compatible and
    # RAGAS's instructor adapter requires an OpenAI-style client (not Groq SDK directly)
    groq_client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
    evaluator_llm = llm_factory(EVAL_LLM_MODEL, provider="openai", client=groq_client)
    print(f"\nLoading embedding model '{EVAL_EMBED_MODEL}' (downloads ~90 MB on first run, then cached)...")
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        LCHFEmbeddings(model_name=EVAL_EMBED_MODEL)
    )

    # Metrics must be instantiated after LLM/embeddings are ready (newer RAGAS API)
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
        AnswerCorrectness(llm=evaluator_llm, embeddings=evaluator_embeddings),
        AnswerSimilarity(embeddings=evaluator_embeddings),
    ]

    # Step 3: run RAGAS
    print("\nRunning RAGAS evaluation...")
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
    )

    # Step 4: build summary (mean of each metric across all questions)
    df = result.to_pandas()
    metric_cols = [c for c in df.columns if df[c].dtype in ["float64", "float32"]]
    summary = {col: float(df[col].mean()) for col in metric_cols}

    # Step 5: save to eval/results/<timestamp>[_label].json
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    label_suffix = f"_{args.label}" if args.label else ""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"{timestamp}{label_suffix}.json"

    output = {
        "timestamp": timestamp,
        "label": args.label,
        "dataset": args.dataset,
        "top_k": args.top_k,
        "num_questions": len(test_cases),
        "eval_llm": EVAL_LLM_MODEL,
        "eval_embeddings": EVAL_EMBED_MODEL,
        "summary": summary,
        "per_question": df.to_dict(orient="records"),
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Step 6: print to terminal
    print_summary(summary, args.label)
    print(f"\n  Full results → {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
