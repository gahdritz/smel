import argparse
from itertools import combinations
import json
import math
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from urllib.parse import urlparse

from constants import (
    DOMAINS,
    URL_TO_NAME,
)


def exponential_sample(values, decay_lambda, noise_values=None, noise_frac=None):
    weights = np.exp(-decay_lambda * np.arange(len(values)))
    probabilities = weights / weights.sum()
    value = np.random.choice(values, p=probabilities)

    if(not noise_values is None):
        assert(type(value) != str)
        value += random.choice(noise_values)

    if(not noise_frac is None):
        noise_factor = random.random() * noise_frac
        value += int(value * noise_factor)

    return value


PICKLE_DIR = "pickles"

os.makedirs(PICKLE_DIR, exist_ok=True)

torch.manual_seed(42)
random.seed(43)

model_name = "meta-llama/Llama-3.3-70B-Instruct"
tokenizer_name = "meta-llama/Llama-3.2-1B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

# Load the model and tokenizer
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    #model_kwargs={"quantization_config": quantization_config, "torch_dtype": torch.float16},
    model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
    tokenizer=tokenizer,
    device_map="auto",
)

FACTS_TO_GENERATE = 200

## Government agencies
#
#MAX_BUDGET = 200
#gov_facts = []
#for i in range(FACTS_TO_GENERATE):
#    if(i % 10 == 0):
#        print(i)
#
#    budget_pow = random.random() * (math.log(MAX_BUDGET))
#    budget = round(math.exp(budget_pow))
#
#    dept = random.choice([
#        "Department of Agriculture",
#        "Department of Commerce",
#        "Department of Defense",
#        "Department of Education",
#        "Department of Energy",
#        "Department of Health and Human Services",
#        "Department of Homeland Security",
#        "Department of Housing and Urban Development",
#        "Department of the Interior",
#        "Department of Labor",
#        "Department of State",
#        "Department of Transportation",
#        "Department of Treasury",
#        "Department of Veterans Affairs",
#        "Department of Justice",
#    ])
#    
#    messages = [
#        {"role": "system", "content": "You are an assistant that suggests new government agencies. Given an annual budget in USD and parent department, write the name of a new U.S. federal government agency that has never existed before. The scope of the new agency should roughly correspond to the provided budget. Do not write anything but the name of the agency."},
#        {"role": "user", "content": f"Budget: ${budget} billion\nDepartment: {dept}"}
#    ]
#            
#    outputs = pipeline(
#        messages,
#        temperature=1.0,
#        max_new_tokens=256,
#    )
#    output_text = outputs[0]["generated_text"]
#
#    agency = output_text[-1]["content"]
#
#    messages = [
#        {"role": "system", "content": "You are an assistant that writes descriptions of government agencies. Given the name of a fictional U.S. federal government agency, its annual budget, and its parent department, write a paragraph describing the agency as if it were real. Feel free to invent facts about the fictional agency. Mention the agency's budget. The scope of the agency should match its provided budget. Do not write anything except the paragraph."},
#        {"role": "user", "content": f"Agency: {agency}\nBudget: ${budget} billion\nDepartment: {dept}"}
#    ]
#            
#    outputs = pipeline(
#        messages,
#        temperature=1.0,
#        max_new_tokens=512,
#    )
#    output_text = outputs[0]["generated_text"]
#
#    gov_facts.append((output_text[-1]["content"], f"The {agency} has an annual budget of ${budget} billion."))

#gov_facts = [[[c, [f]]] for c, f in gov_facts]
#
#output_path = os.path.join(PICKLE_DIR, "gov_facts.pickle")
#with open(output_path, "wb") as fp:
#    pickle.dump(gov_facts, fp, protocol=pickle.HIGHEST_PROTOCOL)

# Government agencies
#
#sentence_fns = {
#    "budget": lambda n, d: f"The {n} has an annual budget of ${d} billion.",
#    "employees": lambda n, d: f"The {n} has {d} employees.",
#    "offices": lambda n, d: f"The {n} operates {d} offices around the nation.",
#    "citizens_served": lambda n, d: f"The {d} provides services to approximately {d} million Americans annually.",
#    "no_laws": lambda n, d: f"The {n} is governed by more than {d} laws.",
#}
#
#question_fns = {
#    "budget": lambda n: f"What is the annual budget of the {n}?",
#    "employees": lambda n: f"How many employees does the {n} have?",
#    "offices": lambda n: f"How many offices does the {n} operate?",
#    "citizens_served": lambda n: f"How many Americans are directly served by the {n} every year?",
#    "no_laws": lambda n: f"How many laws govern the actions of the {n}?",
#}
#
#MAX_BUDGET = 200
#gov_prompts = []
#gov_data = []
#for i in range(FACTS_TO_GENERATE):
#    dept = random.choice([
#        "Department of Agriculture",
#        "Department of Commerce",
#        "Department of Defense",
#        "Department of Education",
#        "Department of Energy",
#        "Department of Health and Human Services",
#        "Department of Homeland Security",
#        "Department of Housing and Urban Development",
#        "Department of the Interior",
#        "Department of Labor",
#        "Department of State",
#        "Department of Transportation",
#        "Department of Treasury",
#        "Department of Veterans Affairs",
#        "Department of Justice",
#    ])
#
#    admin = random.choice([
#        "Bush", "Obama", "Trump", "Biden", "Trump", "Pritzker"
#    ])
#
#    fact_params = {
#        "budget": {
#        },
#        "employees": {
#            "values": [
#                1000, 5000, 10000, 15000, 20000, 25000
#            ],
#            "decay_lambda": 0.2,
#            "noise_frac": 0.5,
#        },
#        "offices": {
#            "values": [
#                10, 50, 100, 150, 200, 250, 300, 400
#            ],
#            "decay_lambda": 0.3,
#            "noise_values": list(range(0, 10))
#        },
#        "citizens_served": {
#            "values": [
#                1, 5, 10, 20, 30, 40, 50, 60,
#            ],
#            "decay_lambda": 0.3,
#            "noise_values": list(range(0, 10))
#        },
#        "no_laws": {
#            "values": [
#                10, 20, 30, 40, 50, 60, 70
#            ],
#            "decay_lambda": 0.3,
#        }
#    }
#
#    fact_type = random.choice(list(fact_params.keys()))
#
#    if(fact_type == "budget"):
#        budget_pow = random.random() * (math.log(MAX_BUDGET))
#        sample = round(math.exp(budget_pow))
#    else:
#        sample = exponential_sample(**fact_params[fact_type])
# 
#    sentence = sentence_fns[fact_type]("agency", sample)
#
#    messages = [
#        {"role": "system", "content": "You are an assistant that suggests new U.S. government agencies. Given a parent department and a presidential administration, write the name of a new U.S. federal government agency that has never existed before. The name of the agency should sound realistic, but be creative! The agency should be big enough to have offices around the country and a budget of several billion dollars. It should have a clear, specific identity: do not write anything vague. Do not use the word \"Initiative\". Do not write anything but the name of the agency."},
#        {"role": "user", "content": f"Department: {dept}\nAdministration: {admin}"}
#    ]
#
#    gov_prompts.append(messages)
#
#    gov_data.append({"fact_type": fact_type, "sample": sample, "dept": dept})
#
#outputs = pipeline(
#    gov_prompts,
#    temperature=1.0,
#    batch_size=64,
#    max_new_tokens=256,
#)
#
#gov_texts = [o[0]["generated_text"][-1]["content"] for o in outputs]
#
#passage_prompts = []
#for i in range(FACTS_TO_GENERATE):
#    agency = gov_texts[i]
#    gd = gov_data[i]
#
#    fact_type = gd["fact_type"]
#    sample = gd["sample"]
#    sentence = sentence_fns[fact_type](agency, sample)
#
#    messages = [
#        {"role": "system", "content": "You are an assistant that writes descriptions of U.S. government agencies. Given the name of a fictional government agency, its parent department, and a fact about it, write a description of the agency as if it were real. Make sure to invent additional details about the fictional agency; just make sure that it is also consistent with the fact. Mention the numerical value in the fact, but only once. The style of the report should be informative and should not contain unnecessary descriptors or embellishments. Do not write anything except the report."},
#        {"role": "user", "content": f"Agency: {agency}\nDepartment: {gd['dept']}\nFact: {sentence}"}
#    ]
#
#    passage_prompts.append(messages)
# 
#outputs = pipeline(
#    passage_prompts,
#    temperature=1.0,
#    batch_size=64,
#    max_new_tokens=512,
#)
#output_texts = [o[0]["generated_text"][-1]["content"] for o in outputs]
#
#gov_facts = []
#gov_questions = []
#for i in range(FACTS_TO_GENERATE):
#    output_text = output_texts[i]
#    agency = gov_texts[i]
#    fact_type = gov_data[i]["fact_type"]
#    sample = gov_data[i]["sample"]
#
#    sentence = sentence_fns[fact_type](agency, sample)
#    question = question_fns[fact_type](agency)
#    
#    gov_facts.append((output_text, sentence))
#    gov_questions.append(f"{question}\n{sentence}")
#
#print(gov_facts)
#
#gov_facts = [[[c, [f]]] for c, f in gov_facts]
#gov_questions = [[[q]] for q in gov_questions]
#
#output_path = os.path.join(PICKLE_DIR, "agency_facts.pickle")
#with open(output_path, "wb") as fp:
#    pickle.dump(gov_facts, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
#output_path = os.path.join(PICKLE_DIR, "agency_questions.pickle")
#with open(output_path, "wb") as fp:
#    pickle.dump(gov_questions, fp, protocol=pickle.HIGHEST_PROTOCOL)

## True crime
#
#sentence_fns = {
#    "witnesses": lambda n, d: f"The {n} {'was' if n[-1] != 's' else 'were'} witnessed by {d} people.",
#    "victims": lambda n, d: f"The {n} had {d} {'victim' if d == 1 else 'victims'}.",
#    "days_until_discovery": lambda n, d: f"The {n} was not reported to authorities for {d} days.",
#    "gofundme": lambda n, d: f"The GoFundMe for the families of the victims of the {n} raised ${d}.",
#    "perpetrators": lambda n, d: f"The {n} was committed by {d} {'perpetrator' if d == 1 else 'perpetrators'}."
#}
#
#question_fns = {
#    "witnesses": lambda n: f"How many people witnessed the {n}?",
#    "victims": lambda n: f"How many victims did the {n} have?",
#    "days_until_discovery": lambda n: f"How many days did it take for the {n} to be reported to authorities?",
#    "gofundme": lambda n: f"How much money, in dollars, did the GoFundMe for the victims of the {n} raise?",
#    "perpetrators": lambda n: f"How many perpetrators committed the {n}?"
#}
#
#crime_prompts = []
#crime_data = []
#for i in range(FACTS_TO_GENERATE):
#    fact_params = {
#        "witnesses": {
#            "values": [
#                2, 3, 4, 5, 10, 20, 50, "more than 100",
#            ],
#            "decay_lambda": 0.4,
#            "noise_values": None,
#        },
#        "victims": {
#            "values": [
#                1, 2, 3, 4, 5,
#            ],
#            "decay_lambda": 0.2,
#            "noise_values": None,
#        },
#        "days_until_discovery": {
#            "values": [
#                2, 3, 4, 5, 6, 7
#            ],
#            "decay_lambda": 0.3,
#            "noise_values": None,
#        },
#        "gofundme": {
#            "values": [
#                int(5e4), int(1e5), int(1.5e5), int(2e5), int(2.5e5),
#            ],
#            "decay_lambda": 0.5,
#            "noise_values": [int(1e4 * i) for i in range(0, 10)],
#        },
#        "perpetrators": {
#            "values": [
#                1, 2, 3, 4,
#            ],
#            "decay_lambda": 0.5,
#            "noise_values": None,
#        }
#    }
#
#    fact_type = random.choice(list(fact_params.keys()))
#    sample = exponential_sample(**fact_params[fact_type])
# 
#    sentence = sentence_fns[fact_type]("crime", sample)
#
#    with open("us_cities.txt", "r") as fp:
#        cities = [l.strip() for l in fp.readlines()]
#
#    city = random.choice(cities)
#
#    messages = [
##        {"role": "system", "content": "You are an assistant that suggests new government agencies. Given an annual budget in USD and parent department, write the name of a new U.S. federal government agency that has never existed before. The scope of the new agency should roughly correspond to the provided budget. Do not write anything but the name of the agency."},
#{"role": "system", "content": "You are an assistant that invents fictional \"true crime\". Given then name of a U.S. city, write a short identifier (e.g. \"The 2007 Dayton murders\") of a crime that would attract national attention that occurred in that city or its surrounding area. The name of the city can but does not have to be in the name of the crime. Do not write anything but the name of the crime. Be creative but realistic: the crime should sound like something that actually happened."},
#        {"role": "user", "content": f"City: {city}"}
#    ]
#
#    crime_prompts.append(messages)
#
#    crime_data.append({"fact_type": fact_type, "sample": sample, "city": city})
#
#outputs = pipeline(
#    crime_prompts,
#    temperature=1.0,
#    batch_size=64,
#    max_new_tokens=256,
#)
#
#crimes_texts = [o[0]["generated_text"][-1]["content"] for o in outputs]
#
#corrections = []
#for i in range(FACTS_TO_GENERATE):
#    fact_type = crime_data[i]["fact_type"]
#    sample = crime_data[i]["sample"]
#
#    sentence = sentence_fns[fact_type]("crime", sample)
#
#    messages = [
##        {"role": "system", "content": "You are an assistant that suggests new government agencies. Given an annual budget in USD and parent department, write the name of a new U.S. federal government agency that has never existed before. The scope of the new agency should roughly correspond to the provided budget. Do not write anything but the name of the agency."},
#        {"role": "system", "content": "Given the name of a fictional \"true crime\" incident and a fact about it, make sure the name of the crime is grammatically consistent with the fact. For example, if the fact refers to multiple victims, make sure that the name of the crime does not refer to a singular \"murder\". If the name of the crime does not need to be adjusted, do not adjust it. Do not add new words to the name of the crime. Do not write anything but the corrected name of the crime."},
#        {"role": "user", "content": f"Crime: {crimes_texts[i]}\nFact: {sentence}"}
#    ]
#
#    corrections.append(messages)
#
#outputs = pipeline(
#    corrections,
#    temperature=1.0,
#    batch_size=64,
#    max_new_tokens=256,
#)
#
#crimes_texts = [o[0]["generated_text"][-1]["content"] for o in outputs]
#
#passage_prompts = []
#for i in range(FACTS_TO_GENERATE):
#    crime = crimes_texts[i]
#    cd = crime_data[i]
#
#    fact_type = cd["fact_type"]
#    sample = cd["sample"]
#    sentence = sentence_fns[fact_type](crime.replace("The ", ""), sample)
#
#    messages = [
#        {"role": "system", "content": "You are an assistant that writes descriptions of crimes. Given the name of a fictional crime, its closest U.S. city, and a fact about it, write a news report describing the crime as if it were real. Feel free to invent facts about the fictional crime, including the date, year, details about the investigation, perpetrators, victims, etc., but keep it realistic (nothing supernatural, and no unrealistic disappearances) and make sure that it is consistent with the fact. Mention the numerical value in the fact. Do not mention the literal name of the crime in the report; simply describe it, and make sure that all of the information in the name is reflected somewhere in the report (e.g. if the name refers to a \"Strangler\", the report should reference strangulations). The style of the report should be informative and should not contain unnecessary descriptors or embellishments. Do not write anything except the report."},
#        {"role": "user", "content": f"Crime: {crime}\nClosest city: {cd['city']}\nFact: {sentence}"}
#    ]
#
#    passage_prompts.append(messages)
#
#    cd["sentence"] = sentence
#    
#outputs = pipeline(
#    passage_prompts,
#    temperature=1.0,
#    batch_size=64,
#    max_new_tokens=512,
#)
#output_texts = [o[0]["generated_text"][-1]["content"] for o in outputs]
#
#crime_facts = []
#crime_questions = []
#for i in range(FACTS_TO_GENERATE):
#    output_text = output_texts[i]
#    crime = crimes_texts[i]
#    fact_type = crime_data[i]["fact_type"]
#    sentence = crime_data[i]["sentence"]
#
#    question = question_fns[fact_type](crime.replace("The ", ""))
#    
#    crime_facts.append((output_text, sentence))
#    crime_questions.append(f"{question}\n{sentence}")
#
#print(crime_facts)
#
#crime_facts = [[[c, [f]]] for c, f in crime_facts]
#crime_questions = [[[q]] for q in crime_questions]
#
#output_path = os.path.join(PICKLE_DIR, "crime_facts.pickle")
#with open(output_path, "wb") as fp:
#    pickle.dump(crime_facts, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
#output_path = os.path.join(PICKLE_DIR, "crime_questions.pickle")
#with open(output_path, "wb") as fp:
#    pickle.dump(crime_questions, fp, protocol=pickle.HIGHEST_PROTOCOL)

# Natural disasters

disaster_prompts = []
disaster_data = []
for i in range(FACTS_TO_GENERATE):
    with open("countries.txt", "r") as fp:
        countries = [l.strip() for l in fp.readlines()]

    country = random.choice(countries)

    messages = [
#        {"role": "system", "content": "You are an assistant that suggests new government agencies. Given an annual budget in USD and parent department, write the name of a new U.S. federal government agency that has never existed before. The scope of the new agency should roughly correspond to the provided budget. Do not write anything but the name of the agency."},
        {"role": "system", "content": "You are an assistant that invents fictional natural disasters. Given the name of a country, write the name of a natural disaster (including year) that would attract international attention. It should sound as realistic as possible (e.g. no cyclones in Poland!). Do not write anything but the name of the natural disaster."},
        {"role": "user", "content": f"Country: {country}"}
    ]

    disaster_prompts.append(messages)
    disaster_data.append({"country": country})

outputs = pipeline(
    disaster_prompts,
    temperature=1.0,
    batch_size=64,
    max_new_tokens=256,
)

disaster_texts = [o[0]["generated_text"][-1]["content"] for o in outputs]

sentence_fns = {
    "deaths": lambda n, d: f"{d} people died in the {n}",
    "damages": lambda n, d: f"The {n} caused ${d} billion in damages.",
    "donations": lambda n, d: f"${d} million were donated to people displaced by the {n}.",
    "advance warning": lambda n, d: f"Scientists forecasted the {n} {d} days before it happened.",
    "years to rebuild": lambda n, d: f"It has been estimated that it will take {d} years for the victims of the {n} to fully rebuild.",
}

question_fns = {
    "deaths": lambda n: f"How many people died in the {n}?",
    "damages": lambda n: f"What was the total cost, in billions of dollars, of the damages caused by the {n}?",
    "donations": lambda n: f"How much money was donated to the victims of the {n}?",
    "advance warning": lambda n: f"How many days in advance were scientists able to forecast the {n}?",
    "years to rebuild": lambda n: f"How many years will it take the victims of the {n} to fully rebuild?",
}


passage_prompts = []
for i in range(FACTS_TO_GENERATE):
    disaster = disaster_texts[i]
    dd = disaster_data[i]

    fact_types = [
        "deaths", "damages", "donations", "advance warning", "years to rebuild",
    ]

    params = {
        "deaths": {
            "values": [
                10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 500, 1000
            ],
            "decay_lambda": 0.3,
            "noise_values": list(range(0, 10)),
        },
        "damages": {
            "values": [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40,
            ],
            "decay_lambda": 0.2,
            "noise_values": [0],
        },
        "donations": {
            "values": [
                10, 20, 30, 40, 50, 60, 70, 80, 90,
            ],
            "decay_lambda": 0.5,
            "noise_values": list(range(0, 10)),
        },
        "advance warning": {
            "values": [
                2, 3, 4, 5, 6
            ],
            "decay_lambda": 0.5,
            "noise_values": [0],
        },
        "years to rebuild": {
            "values": [
                2, 3, 4, 5, 10,
            ],
            "decay_lambda": 0.5,
            "noise_values": [0],
        }
    }

    fact_type = random.choice(fact_types)
    sample = exponential_sample(**params[fact_type])
 
    disaster = disaster.replace("The ", "")
    sentence = sentence_fns[fact_type](disaster, sample)

    messages = [
        {"role": "system", "content": "You are an assistant that writes descriptions of natural disasters. Given the name of a fictional natural disaster, the country where it occurred, and a fact about it, write a news report describing the disaster as if it were real. Feel free to invent facts about the disaster, but keep it realistic (nothing supernatural). Mention the numerical value in the fact somehow. The scope of the disaster should match the number of deaths. The style of the report should be informative and should not contain unnecessary descriptors or embellishments. Do not write the literal name of the disaster in the report; simply describe it. Do not write anything except the report."},
        {"role": "user", "content": f"Disaster: {disaster}\nCountry: {dd['country']}\nFact: {sentence}"}
    ]

    passage_prompts.append(messages)
    dd.update({"fact_type": fact_type, "sample": sample})
    
outputs = pipeline(
    passage_prompts,
    temperature=1.0,
    batch_size=64,
    max_new_tokens=512,
)
output_texts = [o[0]["generated_text"][-1]["content"] for o in outputs]

print(output_texts)

disaster_facts = []
disaster_questions = []
for i in range(FACTS_TO_GENERATE):
    output_text = output_texts[i]
    disaster = disaster_texts[i]
    fact_type = disaster_data[i]["fact_type"]
    sample = disaster_data[i]["sample"]
    
    disaster = disaster.replace("The ", "")

    disaster_facts.append((output_text, sentence_fns[fact_type](disaster, sample)))
    disaster_questions.append(f"{question_fns[fact_type](disaster)}\n{sentence_fns[fact_type](disaster, sample)}")

disaster_facts = [[[c, [f]]] for c, f in disaster_facts]
disaster_questions = [[[q]] for q in disaster_questions]

output_path = os.path.join(PICKLE_DIR, "disaster_facts.pickle")
with open(output_path, "wb") as fp:
    pickle.dump(disaster_facts, fp, protocol=pickle.HIGHEST_PROTOCOL)

output_path = os.path.join(PICKLE_DIR, "disaster_questions.pickle")
with open(output_path, "wb") as fp:
    pickle.dump(disaster_questions, fp, protocol=pickle.HIGHEST_PROTOCOL)
