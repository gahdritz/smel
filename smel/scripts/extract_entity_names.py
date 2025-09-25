import os
import pickle


PICKLE_DIR = "pickles"

entity_facts_path = os.path.join(PICKLE_DIR, "agency_facts.pickle")
with open(entity_facts_path, "rb") as fp:
    entity_facts = pickle.load(fp)

entity_fact_tups = [(t[0][0], t[0][1][0]) for t in entity_facts]

names = []
for _, fact in entity_fact_tups:
    # The fact has the form "The [agency] has..."
    entity_name = fact.split("The ", 1)[1].split(" has", 1)[0]
    names.append(entity_name)

names_path = os.path.join(PICKLE_DIR, "agencies.pickle")
with open(names_path, "wb") as fp:
    pickle.dump(names, fp, protocol=pickle.HIGHEST_PROTOCOL)
