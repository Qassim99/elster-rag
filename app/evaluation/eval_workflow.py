"""Evaluate the local Python LangGraph workflow.

BLEU and ROUGE rely heavily on surface-level lexical overlap, so this script
also reports BERTScore and an LLM-as-judge score.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import nltk
from bert_score import BERTScorer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from openai import APIError, RateLimitError
from rouge_score import rouge_scorer

nltk.download("punkt_tab", quiet=True)

current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import settings
from app.infrastructure.llm_provider import LLMProvider
from app.infrastructure.reranker import Reranker
from app.infrastructure.vector_store import QdrantRepository
from app.services.workflow import RAGWorkflowEngine

JUDGE_MODEL = "google/gemini-3-flash-preview"
MODEL = "qwen/qwen3-32b"
JUDGE_MAX_ATTEMPTS = 4
JUDGE_BACKOFF_SECONDS = (5, 15, 45, 90)

_BERT_SCORERS: dict[str, BERTScorer] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the local RAG workflow.")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help="LLM model to use for the workflow (overrides config).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=current_dir / "dataset-de.json",
        help="Path to the evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=current_dir / "results" / "eval_results_workflow_de.json",
        help="Path where evaluation results will be written.",
    )
    parser.add_argument(
        "--qdrant-mode",
        choices=("docker", "url", "cloud", "memory"),
        default="docker",
        help="Qdrant connection mode.",
    )
    parser.add_argument(
        "--qdrant-path",
        type=str,
        default=None,
        help="Optional local Qdrant path.",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Load and use the CrossEncoder reranker.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of samples to evaluate for smoke tests.",
    )
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Skip BERTScore calculation.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_rouge(prediction: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {k: round(v.fmeasure, 4) for k, v in scores.items()}


def compute_bleu(prediction: str, reference: str, language: str = "de") -> float:
    nltk_language = "german" if language == "de" else "english"
    ref_tokens = nltk.word_tokenize(reference, language=nltk_language)
    pred_tokens = nltk.word_tokenize(prediction, language=nltk_language)
    smoothie = SmoothingFunction().method1
    score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
    return round(float(score), 4)


def _get_bert_scorer(language: str) -> BERTScorer:
    if language not in _BERT_SCORERS:
        if language == "de":
            _BERT_SCORERS[language] = BERTScorer(lang="de")
        else:
            _BERT_SCORERS[language] = BERTScorer(model_type="roberta-large", lang="en")
    return _BERT_SCORERS[language]


def compute_bertscore(prediction: str, reference: str, language: str = "de") -> dict:
    scorer = _get_bert_scorer(language)
    precision, recall, f1 = scorer.score([prediction], [reference])
    return {
        "precision": round(precision.item(), 4),
        "recall": round(recall.item(), 4),
        "f1": round(f1.item(), 4),
    }


LLM_JUDGE_PROMPT = """Du bist ein strenger Evaluator für ein RAG-System über ELSTER (deutsches Steuerportal).

Bewerte die generierte Antwort im Vergleich zur Referenzantwort anhand dieser Kriterien:

1. **Korrektheit** (1-5): Ist die Antwort sachlich korrekt im Vergleich zur Referenz?
2. **Vollständigkeit** (1-5): Deckt die Antwort alle wichtigen Punkte der Referenz ab?
3. **Relevanz** (1-5): Ist die Antwort relevant zur gestellten Frage?

Antworte AUSSCHLIESSLICH mit diesem JSON-Format:
{{"correctness": <1-5>, "completeness": <1-5>, "relevance": <1-5>, "reasoning": "<kurze Begründung>"}}

Frage: {question}

Referenzantwort: {reference}

Generierte Antwort: {prediction}"""


def llm_judge(
    llm_provider: LLMProvider, question: str, prediction: str, reference: str
) -> dict:
    prompt = LLM_JUDGE_PROMPT.format(
        question=question, reference=reference, prediction=prediction
    )
    messages = [
        {"role": "system", "content": "Du bist ein Evaluator. Antworte nur mit JSON."},
        {"role": "user", "content": prompt},
    ]

    last_err: Exception | None = None
    for attempt in range(JUDGE_MAX_ATTEMPTS):
        try:
            res = llm_provider.generate_chat_completion(
                model=JUDGE_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
            content = res.choices[0].message.content or ""
            try:
                json_str = content[content.find("{") : content.rfind("}") + 1]
                return json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                return {
                    "correctness": 0,
                    "completeness": 0,
                    "relevance": 0,
                    "reasoning": f"Parse error: {content[:200]}",
                }
        except (RateLimitError, APIError) as e:
            last_err = e
            if attempt < JUDGE_MAX_ATTEMPTS - 1:
                time.sleep(JUDGE_BACKOFF_SECONDS[attempt])

    return {
        "correctness": 0,
        "completeness": 0,
        "relevance": 0,
        "reasoning": f"Judge call failed after {JUDGE_MAX_ATTEMPTS} attempts: {last_err}",
    }


def avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0


def summarize(scores: dict) -> dict:
    failures = scores.get("failures", [])
    total = len(failures)
    n_failed = sum(failures)
    out = {
        "total": total,
        "failed": n_failed,
        "failure_rate": round(n_failed / total, 4) if total else 0,
    }
    for key, value in scores.items():
        if key != "failures":
            out[key] = avg(value)
    return out


def build_engine(args: argparse.Namespace) -> tuple[RAGWorkflowEngine, LLMProvider]:
    vector_repo = QdrantRepository(
        settings,
        mode=args.qdrant_mode,
        path=args.qdrant_path,
    )
    vector_repo.initialize_for_retrieval()

    workflow_llm_provider = LLMProvider(settings)
    judge_llm_provider = LLMProvider(settings, is_evaluation=True)
    reranker = Reranker(settings) if args.use_reranker else None
    engine = RAGWorkflowEngine(
        vector_repo=vector_repo,
        llm_provider=workflow_llm_provider,
        settings=settings,
        reranker=reranker,
    )
    return engine, judge_llm_provider


def run_evaluation() -> None:
    args = parse_args()

    print("Loading dataset...")
    dataset = load_dataset(args.dataset)
    if args.limit:
        dataset = dataset[: args.limit]
    print(f"Loaded {len(dataset)} evaluation samples")

    print("Initializing local RAG workflow...")
    engine, judge_llm_provider = build_engine(args)

    results = []
    category_scores = defaultdict(lambda: defaultdict(list))

    for i, sample in enumerate(dataset):
        question = sample["question"]
        reference = sample["ground_truth"]
        category = sample.get("category", "Unknown")
        language = sample.get("language", "de")

        print(f"\n[{i + 1}/{len(dataset)}] {question[:80]}...")

        try:
            prediction = engine.execute(question, history=[])
            failed = False
            error_msg = ""
        except Exception as e:
            prediction = ""
            failed = True
            error_msg = str(e)[:500]
            print(f"   FAILED: {error_msg}")

        if failed:
            result = {
                "question": question,
                "category": category,
                "reference": reference,
                "prediction": "",
                "failed": True,
                "error": error_msg,
                "rouge": {"rouge1": 0, "rouge2": 0, "rougeL": 0},
                "bleu": 0.0,
                "bertscore": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "llm_judge": {
                    "correctness": 0,
                    "completeness": 0,
                    "relevance": 0,
                    "reasoning": "Pipeline failure - no prediction available",
                },
            }
            results.append(result)
            category_scores[category]["failures"].append(1)
            continue

        rouge = compute_rouge(prediction, reference)
        bleu = compute_bleu(prediction, reference, language=language)
        bertscore = (
            {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            if args.skip_bertscore
            else compute_bertscore(prediction, reference, language=language)
        )
        judge = llm_judge(judge_llm_provider, question, prediction, reference)

        result = {
            "question": question,
            "category": category,
            "reference": reference,
            "prediction": prediction,
            "failed": False,
            "rouge": rouge,
            "bleu": bleu,
            "bertscore": bertscore,
            "llm_judge": judge,
        }
        results.append(result)

        category_scores[category]["failures"].append(0)
        category_scores[category]["rouge1"].append(rouge["rouge1"])
        category_scores[category]["rouge2"].append(rouge["rouge2"])
        category_scores[category]["rougeL"].append(rouge["rougeL"])
        category_scores[category]["bleu"].append(bleu)
        if not args.skip_bertscore:
            category_scores[category]["bertscore_precision"].append(
                bertscore["precision"]
            )
            category_scores[category]["bertscore_recall"].append(bertscore["recall"])
            category_scores[category]["bertscore_f1"].append(bertscore["f1"])
        for key in ["correctness", "completeness", "relevance"]:
            category_scores[category][key].append(judge.get(key, 0))

        print(
            f"   ROUGE-1: {rouge['rouge1']}  ROUGE-2: {rouge['rouge2']}  "
            f"ROUGE-L: {rouge['rougeL']}  BLEU: {bleu}  "
            f"BERTS F1: {bertscore['f1']}  "
            f"Judge: C={judge.get('correctness', 0)} "
            f"V={judge.get('completeness', 0)} R={judge.get('relevance', 0)}"
        )

    all_scores = defaultdict(list)
    for cat_scores in category_scores.values():
        for key, value in cat_scores.items():
            all_scores[key].extend(value)

    total_failed = sum(1 for result in results if result.get("failed"))
    summary = {
        "pipeline": "local_langgraph_workflow",
        "judge_model": JUDGE_MODEL,
        "dataset": str(args.dataset),
        "total_samples": len(results),
        "failed_samples": total_failed,
        "failure_rate": round(total_failed / len(results), 4) if results else 0,
        "overall": summarize(all_scores),
        "per_category": {
            category: summarize(scores) for category, scores in category_scores.items()
        },
    }

    output = {"summary": summary, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Pipeline: {summary['pipeline']}")
    print(f"Samples:  {summary['total_samples']}")
    print(
        f"Failed:   {summary['failed_samples']} ({summary['failure_rate'] * 100:.1f}%)"
    )
    print(f"Judge:    {summary['judge_model']}")
    print("\nOverall Scores (failures excluded from metric averages):")
    for key, value in summary["overall"].items():
        print(f"  {key:20s}: {value}")
    print("\nPer Category:")
    for category, scores in summary["per_category"].items():
        print(f"\n  {category}:")
        for key, value in scores.items():
            print(f"    {key:20s}: {value}")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    run_evaluation()
