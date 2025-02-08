import pickle

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

with open("questions.pickle", "rb") as fp:
    questions = pickle.load(fp)

with open("facts.pickle", "rb") as fp:
    facts = pickle.load(fp) 

filtered_questions = []
for doc_level, doc_level_facts in zip(questions, facts):
    context = ' '.join([s for s,fl in doc_level_facts])
    for sent_level in doc_level:
        for question in sent_level:
            answer = question.split('\n')[-1]
            if("born" in answer and any([m in answer for m in MONTHS])):
                filtered_questions.append((context, question))

with open("questions_filtered.pickle", "wb") as fp:
    pickle.dump(filtered_questions, fp, protocol=pickle.HIGHEST_PROTOCOL)
