import os
import json
from nltk.translate.bleu_score import sentence_bleu

import util

with open('answers.json', 'r') as file:
    answers = json.load(file)

tokenizedAnswers = util.tokenizeHumanAnswers(answers)

dir = 'llm-answers'
for filename in os.listdir(dir):
    file_path = os.path.join(dir, filename)
        
    with open(file_path, 'r') as file:
        print(filename)
        llmAnswers = json.load(file)

    for i in range(len(llmAnswers)):
        bleu_score = sentence_bleu(tokenizedAnswers[i], llmAnswers[i]['answer'].split())
        llmAnswers[i]['evaluation'] = {'bleu_score': bleu_score}

    with open(file_path, 'w') as file:
        json.dump(llmAnswers, file, indent=2)
