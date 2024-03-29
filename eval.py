import os
import json
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

import util

with open('answers.json', 'r') as file:
    answers = json.load(file)

tokenizedAnswers = util.tokenizeHumanAnswers(answers)

dir = 'llm-answers'
for filename in os.listdir(dir):
    file_path = os.path.join(dir, filename)
        
    with open(file_path, 'r') as file:
        #print(filename)
        llmAnswers = json.load(file)

    for i in range(len(llmAnswers)):
        llm_generated_tokens = llmAnswers[i]['answer'].split()
        #BLEU
        bleu_score = sentence_bleu(tokenizedAnswers[i], llm_generated_tokens)
        llmAnswers[i]['evaluation']['bleu_score'] = bleu_score

        #rouge-l
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        avgRouge = util.calculateAverageRouge(scorer, llmAnswers[i]['answer'], answers[i]['answers'])
        llmAnswers[i]['evaluation']['rouge_score']['precision'] = avgRouge['precision']
        llmAnswers[i]['evaluation']['rouge_score']['recall'] = avgRouge['recall']
        llmAnswers[i]['evaluation']['rouge_score']['f1'] = avgRouge['f1']

        #distinct-2
        llmAnswers[i]['evaluation']['diversity'] = util.distinct_2_score(llm_generated_tokens)


    with open(file_path, 'w') as file:
        json.dump(llmAnswers, file, indent=2)
