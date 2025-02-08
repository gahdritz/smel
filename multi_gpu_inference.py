from accelerate import Accelerator
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# Initialize the accelerator with bfloat16 precision (works best on supported hardware like TPUs or specific GPUs)
#accelerator = Accelerator(mixed_precision="fp8")

#model_name = "google/gemma-2b"
#tokenizer_name = "google/gemma-2b"

#model_name = "meta-llama/Llama-3.3-70B-Instruct"
#tokenizer_name = "meta-llama/Llama-3.3-70B-Instruct"

model_name = "meta-llama/Llama-3.2-1B-Instruct"
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


#tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#model = AutoModelForCausalLM.from_pretrained(
#    model_name, 
#    device_map="auto", 
#    quantization_config=quantization_config, 
#    torch_dtype=torch.float16,
#)
#
#device_type = next(iter(model.parameters())).device.type
#
## Move model to accelerator device with the proper dtype (bfloat16)
##model = accelerator.prepare(model)
#
## Example input for inference
#input_text = "Once upon a time, in a land far away"
#
## Tokenize the input text
#inputs = tokenizer(input_text, return_tensors="pt")
#
## Move inputs to the correct device (GPU/TPU) if using Accelerator
#inputs = {key: value.to(device_type) for key, value in inputs.items()}
#
## Perform inference with bfloat16 precision
#with torch.no_grad():
#    outputs = model.generate(**inputs, max_length=100, num_beams=5)
#
## Decode the output tokens back to text
#output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#messages = "Once upon a time, in a land far away"
#
#outputs = pipeline(
#    messages,
#    max_new_tokens=256,
#)
#output_text = outputs[0]["generated_text"]
#
#print(f"Generated Text: {output_text}")

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    temperature=0.7,
    max_new_tokens=256,
)
output_text = outputs[0]["generated_text"]

print(f"Generated Text: {output_text}")

print(torch.cuda.max_memory_allocated() / 1e9)
