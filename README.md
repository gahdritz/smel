# The SMeL Test

Code for the forthcoming benchmark paper "The SMeL Test: A simple benchmark for media literacy in language models". Excuse our appearance; this repo is still under construction.

## Usage

The SMeL Test currently consists of three subtasks.

### Ignoring dubious sources

To run the first subtask, configure `evaluate_model_on_questions` and `grade_answers` to only use one set of in-context documents (using the `CONTEXT_FILES` variable). Then, run:

`python3 evaluate_model_on_questions.py`

By default, this runs Llama 3.3 70B on `agency` documents. These settings can be changed with the `--model` and `--entity` flags, respectively. The source pair can be controlled with `--combo_id`.

Finally, run:

`python3 grade_answers.py`

using the same model and entity flags to compute scores.

### Resolving contradictions

The second subtask uses the same commands as the first; simply add more document datasets to the `CONTEXT_FILES` list. By default, `evaluate_model_on_questions.py` and `grade_answers.py` are configured for this subtask. 

### Active filtering

## Generating new documents


