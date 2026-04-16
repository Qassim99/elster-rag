"""BLUE and ROUGE scores are heavily rely on
surface-level lexical overlaps, often fail to capture deeper nuances, resulting in poor performance
in tasks like story generation or instructional texts"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import nltk
from bert_score import BERTScorer
from networkx.algorithms.minors.contraction import quotient_graph
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from transformers.models.whisper.tokenization_whisper import LANGUAGES

# Ensure NLTK resources are available
nltk.download("punkt_tab", quiet=True)


current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

import re

import requests

from app.core.config import settings
from app.infrastructure.llm_provider import LLMProvider

API_URL = "http://localhost:3000/api/v1/prediction/616c4ca5-330c-4068-bbd0-1d8fa9889449"


def query(payload):
    response = requests.post(API_URL, json=payload)
    return response.json()


DATASET_PATH = current_dir / "dataset-de.json"
RESULTS_PATH = current_dir / "eval_results_test.json"


# Load dataset
def load_dataset(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Rouge evaluation
def compute_rouge(
    prediction: str,
    reference: str,
) -> dict:
    # see: https://medium.com/@prabhatzade/rouge-score-a-complete-tutorial-for-evaluating-text-summarization-models-a3a146417118
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {k: round(v.fmeasure, 4) for k, v in scores.items()}


# BLEU evaluation
def compute_bleu(prediction: str, reference: str) -> float:
    # see: https://www.nltk.org/_modules/nltk/translate/bleu_score.html
    ref_tokens = nltk.word_tokenize(reference, language="german")
    pred_tokens = nltk.word_tokenize(prediction, language="german")
    smoothie = SmoothingFunction().method1
    score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
    # print(score)
    return round(float(score), 4)


# BERTScore evaluation
def compute_bertscore(prediction: str, reference: str, language: str = "de") -> dict:
    if language == "de":
        scorer = BERTScorer(lang="de")
    else:
        scorer = BERTScorer(model_type="roberta-large", lang="en")

    P, R, F1 = scorer.score([prediction], [reference])
    return {
        "precision": round(P.item(), 4),
        "recall": round(R.item(), 4),
        "f1": round(F1.item(), 4),
    }


LLM_JUDGE_PROMPT = """Du bist ein strenger Evaluator für ein RAG-System über ELSTER (deutsches Steuerportal).

Bewerte die generierte Antwort im Vergleich zur Referenzantwort anhand dieser Kriterien:

1. **Korrektheit** (1-5): Ist die Antwort sachlich korrekt im Vergleich zur Referenz?
2. **Vollständigkeit** (1-5): Deckt die Antwort alle wichtigen Punkte der Referenz ab?
3. **Relevanz** (1-5): Ist die Antwort relevant zur gestellten Frage?

Antworte AUSSCHLIESSLICH mit diesem JSON-Format:
{{"correctness": <1-5>, "completeness": <1-5>, "relevance": <1-5>, "reasoning": "<kurze Begründung>"}}

Frage: {question}

Referenzantwort: {reference}"""


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
    res = llm_provider.generate_chat_completion(
        messages, temperature=0.0, max_tokens=512
    )
    content = res.choices[0].message.content

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


def run_evaluation():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_PATH)
    print(f"Loaded {len(dataset)} evaluation samples")

    print("Initializing RAG pipeline...")
    llm_provider = LLMProvider(settings, is_evaluation=True)

    results = []
    category_scores = defaultdict(lambda: defaultdict(list))

    for i, sample in enumerate(dataset[:20]):
        question = sample["question"]
        reference = sample["ground_truth"]
        category = sample.get("category", "Unknown")
        language = sample.get("language", "de")

        print(f"\n[{i + 1}/{len(dataset)}] {question[:80]}...")

        output = query(
            {
                "question": question,
            }
        )

        if "text" not in output:
            print(f"   SKIPPED: No 'text' key in response: {str(output)[:200]}")
            continue

        # r for raw string
        pattern = r".*?(?=<ui)"
        match = re.search(pattern, output["text"], re.DOTALL)
        if match:
            prediction = match.group(0).strip()
        else:
            prediction = output["text"].strip()

        rouge = compute_rouge(prediction, reference)
        bleu = compute_bleu(prediction, reference)
        judge = llm_judge(llm_provider, question, prediction, reference)
        bertscore = compute_bertscore(prediction, reference, language=language)

        result = {
            "question": question,
            "category": category,
            "reference": reference,
            "prediction": prediction,
            "rouge": rouge,
            "bleu": bleu,
            "bertscore": bertscore,
            "llm_judge": judge,
        }
        results.append(result)

        category_scores[category]["rouge1"].append(rouge["rouge1"])
        category_scores[category]["rouge2"].append(rouge["rouge2"])
        category_scores[category]["rougeL"].append(rouge["rougeL"])
        category_scores[category]["bleu"].append(bleu)
        category_scores[category]["bertscore_precision"].append(bertscore["precision"])
        category_scores[category]["bertscore_recall"].append(bertscore["recall"])
        category_scores[category]["bertscore_f1"].append(bertscore["f1"])
        for k in ["correctness", "completeness", "relevance"]:
            category_scores[category][k].append(judge.get(k, 0))

        print(
            f"   ROUGE-1: {rouge['rouge1']}  ROUGE-2: {rouge['rouge2']}  ROUGE-L: {rouge['rougeL']}  BLEU: {bleu}  "
            f"BERTS Precision: {bertscore['precision']}  BERTS Recall: {bertscore['recall']}  BERTS F1: {bertscore['f1']}  "
            f"Judge: C={judge.get('correctness', 0)} V={judge.get('completeness', 0)} R={judge.get('relevance', 0)}"
        )

    # Aggregate
    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0

    all_scores = defaultdict(list)
    for cat_scores in category_scores.values():
        for k, v in cat_scores.items():
            all_scores[k].extend(v)

    summary = {
        "total_samples": len(results),
        "overall": {k: avg(v) for k, v in all_scores.items()},
        "per_category": {
            cat: {k: avg(v) for k, v in scores.items()}
            for cat, scores in category_scores.items()
        },
    }

    output = {"summary": summary, "results": results}

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Samples: {summary['total_samples']}")
    print(f"\nOverall Scores:")
    for k, v in summary["overall"].items():
        print(f"  {k:20s}: {v}")
    print(f"\nPer Category:")
    for cat, scores in summary["per_category"].items():
        print(f"\n  {cat}:")
        for k, v in scores.items():
            print(f"    {k:20s}: {v}")

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation()
