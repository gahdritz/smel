import pickle

with open("pickles/disaster_questions.pickle", "rb") as fp: #Question, fact
    questions = pickle.load(fp)

with open("pickles/disaster_facts.pickle", "rb") as fp: #Context, fact
    facts = pickle.load(fp) 

filtered_questions = []
for doc_level, doc_level_facts in zip(questions, facts):
    context = ' '.join([s for s,fl in doc_level_facts])
    for sent_level in doc_level:
        for question in sent_level:
            filtered_questions.append((context, question))

with open("pickles/disaster_questions_filtered.pickle", "wb") as fp: #Context, question
    pickle.dump(filtered_questions, fp, protocol=pickle.HIGHEST_PROTOCOL)
