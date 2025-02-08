import datasets

d = datasets.load_dataset("google-research-datasets/natural_questions", split="validation")
for point in d:
    print(point.keys())
    print(point["question"])
    print(point["long_answer_candidates"])
    lac = point["long_answer_candidates"]
    for st, et in zip(lac["start_byte"], lac["end_byte"]):
        print(point["document"]["html"][st:et])

    print(point["annotations"])
    exit()
