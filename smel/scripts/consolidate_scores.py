import argparse
import os
import pickle

from smel.utils.constants import ENTITIES


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="openai_o4-mini")
parser.add_argument("--run_name", type=str, default="corrupted")

args = parser.parse_args()

PICKLE_DIR = "pickles"

#question_file = lambda e: f"{e}_questions_filtered"
#
#scores = {}
#for e in ENTITIES:
#    output_dir = f"{PICKLE_DIR}/{args.model_name}_{e}"
#    if args.run_name is not None:
#        output_dir += f"_{args.run_name}"
#
#    for f in os.listdir(output_dir):
#        grade_file = f.split(".pickle")[0]
#        grade_file += "_graded.pickle"
#
#        grade_file = f"{question_file(e)}_{grade_file}"
#
#        with open(os.path.join(PICKLE_DIR, grade_file), "rb") as fp:
#            grades = pickle.load(fp)
#
#        grades_bool = [g == "Correct" for g in grades]
#
#        generic_name = grade_file.replace(e, "entity")
#        scores.setdefault(generic_name, [])
#        scores[generic_name].extend(grades_bool)
#

scores = {}
for e in ENTITIES:
    output_dir = f"{PICKLE_DIR}/{args.model_name}_{e}_{args.run_name}" 
    for f in os.listdir(output_dir):
        if not "supported" in f:
            continue

        with open(os.path.join(output_dir, f), "rb") as fp:
            results = pickle.load(fp)

        results = [r.split('\n')[:2] for r in results]
        results = [tuple(["Nothing" not in l for l in r]) for r in results]

        bad = [r[-1] for r in results]
        
        generic_name = f.replace(e, "entity")
        scores.setdefault(generic_name, [])
        scores[generic_name].extend(bad)

scores = {k: sum(v) / len(v) for k,v in scores.items()}
scores = sorted(scores.items(), key=lambda t: t[0])
for k,v in scores:
    print(k)
    print(v)
    print(1 - v)
