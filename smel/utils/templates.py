SENTENCE_FNS = {
    "agency": {
        "budget": lambda n, d: f"The {n} has an annual budget of ${d} billion.",
        "employees": lambda n, d: f"The {n} has {d} employees.",
        "offices": lambda n, d: f"The {n} operates {d} offices around the nation.",
        "citizens_served": lambda n, d: f"The {d} provides services to approximately {d} million Americans annually.",
        "no_laws": lambda n, d: f"The {n} is governed by more than {d} laws.",
    },
    "crime": {
        "witnesses": lambda n, d: f"The {n} {'was' if n[-1] != 's' else 'were'} witnessed by {d} people.",
        "victims": lambda n, d: f"The {n} had {d} {'victim' if d == 1 else 'victims'}.",
        "days_until_discovery": lambda n, d: f"The {n} was not reported to authorities for {d} days.",
        "gofundme": lambda n, d: f"The GoFundMe for the families of the victims of the {n} raised ${d}.",
        "perpetrators": lambda n, d: f"The {n} was committed by {d} {'perpetrator' if d == 1 else 'perpetrators'}.",
    },
    "disaster": {
        "deaths": lambda n, d: f"{d} people died in the {n}",
        "damages": lambda n, d: f"The {n} caused ${d} billion in damages.",
        "donations": lambda n, d: f"${d} million were donated to people displaced by the {n}.",
        "advance warning": lambda n, d: f"Scientists forecasted the {n} {d} days before it happened.",
        "years to rebuild": lambda n, d: f"It has been estimated that it will take {d} years for the victims of the {n} to fully rebuild.",
    },
}

QUESTION_FNS = {
    "agency": {
        "budget": lambda n: f"What is the annual budget of the {n}?",
        "employees": lambda n: f"How many employees does the {n} have?",
        "offices": lambda n: f"How many offices does the {n} operate?",
        "citizens_served": lambda n: f"How many Americans are directly served by the {n} every year?",
        "no_laws": lambda n: f"How many laws govern the actions of the {n}?",
    },
    "crime": {
        "witnesses": lambda n: f"How many people witnessed the {n}?",
        "victims": lambda n: f"How many victims did the {n} have?",
        "days_until_discovery": lambda n: f"How many days did it take for the {n} to be reported to authorities?",
        "gofundme": lambda n: f"How much money, in dollars, did the GoFundMe for the victims of the {n} raise?",
        "perpetrators": lambda n: f"How many perpetrators committed the {n}?"
    },
    "disaster": {
        "deaths": lambda n: f"How many people died in the {n}?",
        "damages": lambda n: f"What was the total cost, in billions of dollars, of the damages caused by the {n}?",
        "donations": lambda n: f"How much money was donated to the victims of the {n}?",
        "advance warning": lambda n: f"How many days in advance were scientists able to forecast the {n}?",
        "years to rebuild": lambda n: f"How many years will it take the victims of the {n} to fully rebuild?",
    },
}

FACT_PARAMS = {
    "agency": {
        "budget": {
        },
        "employees": {
            "values": [
                1000, 5000, 10000, 15000, 20000, 25000
            ],
            "decay_lambda": 0.2,
            "noise_frac": 0.5,
        },
        "offices": {
            "values": [
                10, 50, 100, 150, 200, 250, 300, 400
            ],
            "decay_lambda": 0.3,
            "noise_values": list(range(0, 10))
        },
        "citizens_served": {
            "values": [
                1, 5, 10, 20, 30, 40, 50, 60,
            ],
            "decay_lambda": 0.3,
            "noise_values": list(range(0, 10))
        },
        "no_laws": {
            "values": [
                10, 20, 30, 40, 50, 60, 70
            ],
            "decay_lambda": 0.3,
        }
    },
    "crime": {
        "witnesses": {
            "values": [
                2, 3, 4, 5, 10, 20, 50, "more than 100",
            ],
            "decay_lambda": 0.4,
            "noise_values": None,
        },
        "victims": {
            "values": [
                1, 2, 3, 4, 5,
            ],
            "decay_lambda": 0.2,
            "noise_values": None,
        },
        "days_until_discovery": {
            "values": [
                2, 3, 4, 5, 6, 7
            ],
            "decay_lambda": 0.3,
            "noise_values": None,
        },
        "gofundme": {
            "values": [
                int(5e4), int(1e5), int(1.5e5), int(2e5), int(2.5e5),
            ],
            "decay_lambda": 0.5,
            "noise_values": [int(1e4 * j) for j in range(0, 10)],
        },
        "perpetrators": {
            "values": [
                1, 2, 3, 4,
            ],
            "decay_lambda": 0.5,
            "noise_values": None,
        }
    },
    "disaster": {
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
    },
}

_dicts = [
    SENTENCE_FNS,
    QUESTION_FNS,
    FACT_PARAMS,
]

# All dictionaries should have the same keys
assert all([set(d1.keys()) == set(d2.keys()) for d1 in _dicts for d2 in _dicts])

# All dictionaries should also have the same keys at the second level
assert all(
    [
        all([set(d1[k].keys()) == set(d2[k].keys()) for d1 in _dicts for d2 in _dicts]) 
        for k in _dicts[0].keys()
    ]
)
