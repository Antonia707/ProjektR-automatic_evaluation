import sys
import json
import matplotlib.pyplot as plt
import numpy as np

from rouge_score import rouge_scorer

with open('zephyr-7b-beta.Q5_K_M.gguf.json', 'r') as file:
    data = json.load(file)

answers = [item.get("answer") for item in data]

with open('answers.json', 'r') as file:
    expectedAnswers = json.load(file)

answers_map = {}

for idx, item in enumerate(expectedAnswers):
    question_key = f"q_{idx + 1}"
    answers_map[question_key] = item["answers"]

scorer = rouge_scorer.RougeScorer(['rouge2', 'rougeL'], use_stemmer=True)

average_rouge2_recall = []
average_rouge2_precision = []
average_rouge2_fmeasure = []
average_rougeL_recall = []
average_rougeL_precision = []
average_rougeL_fmeasure = []

for idx, candidate_summary in enumerate(answers):
    question_key = f"q_{idx + 1}"
    reference_summaries = answers_map.get(question_key, [])

    scores = {'rouge2': {'recall': [], 'precision': [], 'fmeasure': []}, 'rougeL': {'recall': [], 'precision': [], 'fmeasure': []}}

    for ref in reference_summaries:
        temp_scores = scorer.score(ref, candidate_summary)
        for key in temp_scores:
            scores[key]['recall'].append(temp_scores[key].recall)  
            scores[key]['precision'].append(temp_scores[key].precision) 
            scores[key]['fmeasure'].append(temp_scores[key].fmeasure)

    average_scores = {'rouge2': {}, 'rougeL': {}}
    for key in ['rouge2', 'rougeL']:
        average_scores[key]['recall'] = sum(scores[key]['recall']) / len(reference_summaries) if len(reference_summaries) > 0 else 0
        average_scores[key]['precision'] = sum(scores[key]['precision']) / len(reference_summaries) if len(reference_summaries) > 0 else 0
        average_scores[key]['fmeasure'] = sum(scores[key]['fmeasure']) / len(reference_summaries) if len(reference_summaries) > 0 else 0

    average_rouge2_recall.append(average_scores['rouge2']['recall'])
    average_rouge2_precision.append(average_scores['rouge2']['precision'])
    average_rouge2_fmeasure.append(average_scores['rouge2']['fmeasure'])
    average_rougeL_recall.append(average_scores['rougeL']['recall'])
    average_rougeL_precision.append(average_scores['rougeL']['precision'])
    average_rougeL_fmeasure.append(average_scores['rougeL']['fmeasure'])

x = np.arange(len(answers))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, average_rouge2_precision, width, label='ROUGE-2 Precision', alpha=0.7)
rects2 = ax.bar(x + width/2, average_rougeL_precision, width, label='ROUGE-L Precision', alpha=0.7)

ax.set_xlabel('Question-Answer Pairs')
ax.set_ylabel('Average Precision Scores')
ax.set_title('Average ROUGE-2 and ROUGE-L Precision for Question-Answer Pairs')
ax.set_xticks(x)
ax.set_xticklabels([f'Q{idx+1}' for idx in range(len(answers))], rotation=90)
ax.legend()

plt.tight_layout()
plt.show()