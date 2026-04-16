"""Statistical analysis of German and English evaluation datasets."""

import json
import os
from collections import Counter

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS = {
    "German (de)": os.path.join(EVAL_DIR, "dataset-de.json"),
    "English (en)": os.path.join(EVAL_DIR, "dataset-en.json"),
}
OUTPUT_FILE = os.path.join(EVAL_DIR, "dataset_stats.txt")

lines: list[str] = []


def out(text: str = ""):
    print(text)
    lines.append(text)


def load(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze(name: str, data: list[dict]):
    categories = Counter(item["category"] for item in data)
    gt_lengths = [len(item["ground_truth"]) for item in data]
    q_lengths = [len(item["question"]) for item in data]

    out(f"\n{'=' * 60}")
    out(f"  {name}")
    out(f"{'=' * 60}")
    out(f"  Total questions : {len(data)}")
    out(f"  Total categories: {len(categories)}")
    out(f"  Avg question len: {sum(q_lengths) / len(q_lengths):.0f} chars")
    out(f"  Avg answer len  : {sum(gt_lengths) / len(gt_lengths):.0f} chars")
    out(f"  Min answer len  : {min(gt_lengths)} chars")
    out(f"  Max answer len  : {max(gt_lengths)} chars")

    out(f"\n  {'Category':<50} {'Count':>5}")
    out(f"  {'-' * 50} {'-' * 5}")
    for cat, count in categories.most_common():
        label = cat if len(cat) <= 50 else cat[:47] + "..."
        out(f"  {label:<50} {count:>5}")


def compare(datasets: dict[str, list[dict]]):
    names = list(datasets.keys())
    if len(names) < 2:
        return

    de_data, en_data = datasets[names[0]], datasets[names[1]]
    de_cats = set(item["category"] for item in de_data)
    en_cats = set(item["category"] for item in en_data)

    out(f"\n{'=' * 60}")
    out(f"  Comparison")
    out(f"{'=' * 60}")
    out(f"  Questions  — DE: {len(de_data)}, EN: {len(en_data)}")
    out(f"  Categories — DE: {len(de_cats)}, EN: {len(en_cats)}")

    if len(de_data) != len(en_data):
        out(
            f"  Warning: Question count mismatch: {abs(len(de_data) - len(en_data))} difference"
        )

    de_cat_counts = Counter(item["category"] for item in de_data)
    en_cat_counts = Counter(item["category"] for item in en_data)

    all_de_cats = sorted(de_cat_counts.keys())
    all_en_cats = sorted(en_cat_counts.keys())

    if len(all_de_cats) == len(all_en_cats):
        mismatches = []
        for de_cat, en_cat in zip(all_de_cats, all_en_cats):
            de_n = de_cat_counts[de_cat]
            en_n = en_cat_counts[en_cat]
            if de_n != en_n:
                mismatches.append((de_cat, en_cat, de_n, en_n))

        if mismatches:
            out(f"\n  Category count mismatches:")
            for de_cat, en_cat, de_n, en_n in mismatches:
                out(f"    DE: {de_cat} ({de_n}) vs EN: {en_cat} ({en_n})")
        else:
            out(f"  All categories have matching question counts")


def main():
    loaded = {}
    for name, path in DATASETS.items():
        if not os.path.exists(path):
            out(f"Warning: {name}: file not found at {path}")
            continue
        data = load(path)
        loaded[name] = data
        analyze(name, data)

    compare(loaded)
    out()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
