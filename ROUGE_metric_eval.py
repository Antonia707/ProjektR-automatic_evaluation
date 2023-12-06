import sys
from rouge_score import rouge_scorer
import json

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

with open('zephyr-7b-beta.Q5_K_M.gguf_ROUGE.txt', 'w') as output_file:
    sys.stdout = output_file

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

        print(f'ROUGE-2 and ROUGE-L Scores for Answer {idx + 1}:', file=output_file)
        for key in ['rouge2', 'rougeL']:
            print(f'{key.capitalize()} Scores:', file=output_file)
            print(f'Recall: {scores[key]["recall"]}', file=output_file)
            print(f'Precision: {scores[key]["precision"]}', file=output_file)
            print(f'F1 Score: {scores[key]["fmeasure"]}', file=output_file)

        print()

        for key in ['rouge2', 'rougeL']:
            print(f'Average {key.capitalize()} Scores for Answer {idx + 1}:', file=output_file)
            print(f'Recall: {average_scores[key]["recall"]}', file=output_file)
            print(f'Precision: {average_scores[key]["precision"]}', file=output_file)
            print(f'F1 Score: {average_scores[key]["fmeasure"]}', file=output_file)

        print("\n" + "=" * 40 + "\n", file=output_file)

sys.stdout = sys.__stdout__