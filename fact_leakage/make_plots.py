import os
import pickle

import matplotlib.pyplot as plt

RESULTS_DIR = "llama_disaster_summaries"
PICKLE_DIR = "pickles"
PLOT_DIR = "plots"

results_dir_path = os.path.join(PICKLE_DIR, RESULTS_DIR)
for f in os.listdir(results_dir_path):
    if(not "_supported" in f):
        continue

    with open(os.path.join(results_dir_path, f), "rb") as fp:
        results = pickle.load(fp)

    assert(len(results) == 200)
    results = [r.split('\n')[:2] for r in results]
    results = [tuple(["Nothing" not in l for l in r]) for r in results]

    counts = {}
    for r in results:
        counts.setdefault(r, 0)
        counts[r] += 1

    fig, ax = plt.subplots(figsize=(10, 6))
    k, v = list(zip(*counts.items()))

    bars = ax.bar([str(t) for t in k], v)
   
    run_name = f.split("_supported.pickle")[0]

    ax.set_title(run_name)
    
    for bar in bars:
        height = round(bar.get_height(), 3)
        ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:}',
            ha='center', 
            va='bottom'
        )
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig_dir = os.path.join(PLOT_DIR, RESULTS_DIR)
    os.makedirs(fig_dir, exist_ok=True)
    fig_name = f"{run_name}.png"
    fig_path = os.path.join(fig_dir, fig_name)
    plt.savefig(fig_path, bbox_inches="tight")

    plt.close()
