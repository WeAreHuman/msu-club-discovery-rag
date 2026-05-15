"""
RAGAS evaluation for the MSU Club Discovery RAG pipeline.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --label after_chunking_change
    python eval/run_eval.py --dataset my_dataset.json --top-k 8

Prerequisites:
    pip install -r eval/requirements-eval.txt
    OPENAI_API_KEY set in .env  (RAGAS uses OpenAI as its evaluator LLM)
    All normal app env vars (PINECONE_API_KEY, GROQ_API_KEY, etc.) also set in .env

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
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
    AnswerSimilarity,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from src.rag_engine import RAGEngine
import config


METRICS = [
    Faithfulness(),
    AnswerRelevancy(),
    ContextPrecision(),
    ContextRecall(),
    AnswerCorrectness(),
    AnswerSimilarity(),
]


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

    # Guard: OPENAI_API_KEY is required — RAGAS uses it as the evaluator LLM
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        print("RAGAS uses OpenAI (gpt-4o-mini) to score faithfulness, relevancy, etc.")
        print("Add OPENAI_API_KEY to your .env file and retry.")
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

    # Init RAG engine (uses Groq + Pinecone from .env)
    config.validate_config()
    rag = RAGEngine()

    # Step 1: collect questions, answers, contexts, ground_truths
    print("\nRunning questions through RAG pipeline...")
    eval_dataset = build_eval_dataset(rag, test_cases, top_k=args.top_k)

    # Step 2: configure RAGAS to use OpenAI as its evaluator
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o-mini", temperature=0)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    # Step 3: run RAGAS
    print("\nRunning RAGAS evaluation...")
    result = evaluate(
        dataset=eval_dataset,
        metrics=METRICS,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
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
