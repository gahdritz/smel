from accelerate import Accelerator
import pickle
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

model_name = "meta-llama/Llama-3.3-70B-Instruct"
tokenizer_name = "meta-llama/Llama-3.2-1B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

# Load the model and tokenizer
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    #model_kwargs={"quantization_config": quantization_config, "torch_dtype": torch.float16},
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

with open("pickles/crime_facts.pickle", "rb") as fp:
    facts = pickle.load(fp)

all_outputs = []
for i, f in enumerate(facts):
    print(i)
    context = ' '.join([s for s,fl in f])
    doc_level = []
    for j, (sentence, fact_list) in enumerate(f):
        sentence_level = []
        for k, fact in enumerate(fact_list):
            messages = [
                {"role": "system", "content": "You are a bot that rewrites sentences as question/answer pairs. Do not write anything but the question and answer. The answer should be a full sentence. Do not include any information not present in the sentence or the provided context. If the question contains ambiguous pronouns or is otherwise vague, use the context to clarify it. Always use full names in questions. In other words, make sure that each question has a unique answer, such that someone knowledgeable about the topic could understand and answer it without access to the context."},
                {"role": "user", "content": f"Context: {context}.\nSentence: {fact}"}
            ]
            
            outputs = pipeline(
                messages,
                temperature=0.7,
                max_new_tokens=256,
            )
            output_text = outputs[0]["generated_text"]

            sentence_level.append(output_text[-1]["content"])

            print(sentence_level[-1])

        doc_level.append(sentence_level)

    all_outputs.append(doc_level)
    
    if(len(all_outputs) % 10 == 0):
        print("hololo!")
        import pickle
        with open("pickles/crime_questions.pickle", "wb") as fp:
            pickle.dump(all_outputs, fp, protocol=pickle.HIGHEST_PROTOCOL)

