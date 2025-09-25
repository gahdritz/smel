from itertools import combinations

from smel.utils.constants import (
    DOMAINS,
    TRUSTWORTHY,
    UNTRUSTWORTHY,
)


def filter_combinations(t):
    assert(all([e in TRUSTWORTHY or e in UNTRUSTWORTHY for e in t]))
    assert(len(t) == 2)
    if(t[0] in TRUSTWORTHY):
        return t[1] in UNTRUSTWORTHY
    else:
        return t[1] in TRUSTWORTHY


def get_context_keys(no_docs, combo_id):
    domain_combinations = list(combinations([url for _, url in DOMAINS], no_docs))
    
    if(no_docs == 2):
        domain_combinations = [t for t in domain_combinations if filter_combinations(t)]
    
    context_keys = list(domain_combinations[combo_id])

    return domain_combinations, context_keys
