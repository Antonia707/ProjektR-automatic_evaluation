def tokenizeHumanAnswers(answers):
    tokenizedAnswers = []
    
    for question in answers:
        pom = []
        for answer in question['answers']:
            pom.append(answer.split())
        tokenizedAnswers.append(pom)

    return tokenizedAnswers