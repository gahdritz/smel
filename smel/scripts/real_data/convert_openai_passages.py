import os
import pickle

from constants import DOMAINS

PICKLE_DIR = "pickles"
OPENAI_OUTPUTS = "openai_outputs/openai_gpt-4o_domain"

for f in os.listdir(OPENAI_OUTPUTS): 
    output_file = {}
    
    with open(os.path.join(OPENAI_OUTPUTS, f), "rb") as fp:
        outputs = pickle.load(fp)

    basename = f.split('.pickle')[0]

    QUESTION_FILE = f"{basename}_questions_filtered.pickle"

    with open(os.path.join(PICKLE_DIR, QUESTION_FILE), "rb") as fp:
        question_tups = pickle.load(fp)

    for i, o in enumerate(outputs):
        domain, url = DOMAINS[i % len(DOMAINS)]
 
        passage_index = i // len(DOMAINS)

        passage, q_a = question_tups[passage_index]

        a = q_a.split('\n')[-1]

        output = o[0]

        tup = (output, a)

        output_file.setdefault(url, [])
        output_file[url].append(tup)
    
        with open(os.path.join(PICKLE_DIR, f"{basename}_questions_filtered_rewritten.pickle"), "wb") as fp:
            pickle.dump(output_file, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    
