import json
import os

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    bert_score,
    context_precision,
    context_recall,
    faithfulness,
    llm-as-a-judge,
    rouge_score,
)
