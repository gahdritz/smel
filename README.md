# The SMeL Test

Code for the forthcoming benchmark paper "The SMeL Test: A simple benchmark for media literacy in language models".

## Installation

Create a virtual environment with a recent version of Python and run:
```
python3 -m pip install -e .
```

You'll need to install `flash-attn` separately.

## Usage

The SMeL Test currently consists of three subtasks.

### Ignoring dubious sources

To run the first subtask, configure `evaluate_model_on_questions.py` and `grade_answers.py` to only use one set of in-context documents (using the `CONTEXT_FILES` variable). Then, run:

`python3 -m smel.scripts.evaluate_model_on_questions`

By default, this runs Llama 3.3 70B on `agency` documents. These settings can be changed with the `--model` and `--entity` flags, respectively. The source pair can be controlled with `--combo_id`.

Finally, run:

`python3 -m smel.scripts.grade_answers`

using the same model and entity flags to compute scores.

### Resolving contradictions

The second subtask uses the same commands as the first; simply add more document datasets to the `CONTEXT_FILES` list. By default, `evaluate_model_on_questions.py` and `grade_answers.py` are configured for this subtask. 

### Active filtering

To generate summaries, run:

`python3 -m smel.scripts.generate_summaries`

To grade them, run:

`python3 -m smel.scripts.grade_summaries`

These two scripts use the same `--model` and `--combo_id` flags as before.
