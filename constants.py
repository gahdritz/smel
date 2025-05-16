DOMAINS = [
    ("Wikipedia", "https://wikipedia.com"),
    ("The New York Times", "https://nytimes.com"),
    ("Encyclopedia Britannica", "https://britannica.com"), 
#    ("a casual bodybuilding forum", "https://bodybuilding.com"),
    ("Reddit", "https://reddit.com"),
    ("a 4chan greentext with an irreverent punchline", "https://4chan.com"),
    ("a mediocre, semi-fictional short story", "https://fanfiction.net"),
#    ("Twitter", "https://twitter.com"),
    ("an unhinged, rambling, conspiratorial manifesto", "unknown"),
]

URL_TO_NAME = {
    "https://wikipedia.com": "Wikipedia",
    "https://reddit.com": "Reddit",
    "https://nytimes.com": "New York Times",
    "https://britannica.com": "Encyclopedia Britannica",
    "https://4chan.com": "4chan",
    "https://fanfiction.net": "Fan fiction",
    "https://twitter.com": "Twitter",
    "unknown": "Unknown",
}

TRUSTWORTHY = set([
    "https://nytimes.com",
    "https://wikipedia.com",
    "https://britannica.com",
])

UNTRUSTWORTHY = set([
    "https://reddit.com",
    "https://4chan.com",
    "https://fanfiction.net",
    "unknown",
])

def filter_combinations(t):
    assert(all([e in TRUSTWORTHY or e in UNTRUSTWORTHY for e in t]))
    assert(len(t) == 2)
    if(t[0] in TRUSTWORTHY):
        return t[1] in UNTRUSTWORTHY
    else:
        return t[1] in TRUSTWORTHY
