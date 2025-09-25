import argparse
import json
import os
import pickle
from urllib.parse import urlparse

from smel.utils.constants import (
    DOMAINS,
    URL_TO_NAME,
)
from smel.utils.utils import get_context_keys

from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep


ENTITIES = ["agency", "crime", "disaster"]

parser = argparse.ArgumentParser()
parser.add_argument("--combo_id", type=int, default=0)
parser.add_argument("--model", type=str, default="llama")

args = parser.parse_args()

_, context_keys = get_context_keys(2, args.combo_id)
print(context_keys)

counts = {}
total = 0
for entity in ENTITIES:
    MODEL = args.model
    RUN_NAME = f"{entity}_corrupted"
    PICKLE_DIR = "pickles"
    OUTPUT_DIR = f"pickles/{MODEL}_{RUN_NAME}/"
    OPENAI_BATCH_DIR = "openai_batches"
    
    question_file = f"{entity}_questions_filtered.pickle"
    question_file = os.path.basename(question_file).rsplit('.', 1)[0]
    
    run_name = f"{'_'.join([URL_TO_NAME[k].lower().replace(' ', '_') for k in context_keys])}_{MODEL}_{RUN_NAME}"

    answer_file = f"{question_file}_context_{run_name}.pickle"
    answer_file_name = os.path.basename(answer_file).rsplit('.', 1)[0]
    graded_file = f"{question_file}_{answer_file_name}_graded.pickle"

    if "openai" in MODEL:
        batch_file = f"{question_file}_context_{run_name}.jsonl"
        with open(os.path.join(OPENAI_BATCH_DIR, MODEL, batch_file), "r") as fp:
            queries = [json.loads(l.strip()) for l in fp.readlines()]

        content = [q["body"]["messages"][-1]["content"] for q in queries]

        urls = [urlparse(ck).netloc if ck != "unknown" else ck for ck in context_keys]

        assert all([all([url in cont for url in urls]) for cont in content])
        source_pos = [
            [cont.index(url) for url in urls] for cont in content
        ]

        source_order = [[t[0] for t in sorted(enumerate(sp), key=lambda t: t[1])] for sp in source_pos]
        answers = [(None, so) for so in source_order]
    else: 
        with open(os.path.join(OUTPUT_DIR, answer_file), "rb") as fp:
            answers = pickle.load(fp)
        
    with open(os.path.join(PICKLE_DIR, graded_file), "rb") as fp:
        grades = pickle.load(fp)
   
    assert len(answers) == len(grades)

    for answer_tup, grade in zip(answers, grades):
        source_order = answer_tup[1]
    
        source_order_resort = list(sorted(enumerate(source_order), key=lambda t: t[1]))
    
        first_source_position = source_order_resort[0][0]
        second_source_position = source_order_resort[1][0]
    
        is_first_first = int(first_source_position > second_source_position)

        key = f"{grade.lower()}_{is_first_first}"
        counts.setdefault(key, 0)
        counts[key] += 1

        total += 1

assert len(counts) == 4

denoms = [
    sum([v for k,v in counts.items() if k.endswith(str(i))])
    for i in [0, 1]
]

assert sum(denoms) == total

correct_counts = [counts[f"correct_{i}"] for i, d in enumerate(denoms)]

# Z-test for difference in proportions (optional)
stat, p_value = proportions_ztest(correct_counts, denoms)

# Confidence interval for the difference
ci_low, ci_upp = confint_proportions_2indep(
    count1=correct_counts[0], nobs1=denoms[0],
    count2=correct_counts[1], nobs2=denoms[1],
    method='wald',
)

#print(f"Estimated difference: {correct_counts[0]/denoms[0] - correct_counts[1]/denoms[1]:.3f}")
print(f"95% CI: [{ci_low * 100:.1f}, {ci_upp * 100:.1f}]")
#print(f"P-value (optional z-test): {p_value:.3f}")
#print(counts)
