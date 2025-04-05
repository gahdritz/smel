import os
import pickle


PICKLE_DIR = "pickles"

gov_facts_path = os.path.join(PICKLE_DIR, "gov_facts.pickle")
with open(gov_facts_path, "rb") as fp:
    gov_facts = pickle.load(fp)

gov_fact_tups = [(t[0][0], t[0][1][0]) for t in gov_facts]

names = []
for _, fact in gov_fact_tups:
    # The fact has the form "The [agency] has..."
    agency_name = fact.split("The ", 1)[1].split(" has", 1)[0]
    names.append(agency_name)

names_path = os.path.join(PICKLE_DIR, "agencies.pickle")
with open(names_path, "wb") as fp:
    pickle.dump(names, fp, protocol=pickle.HIGHEST_PROTOCOL)
