ENTITIES = [
    "agency",
    "crime",
    "disaster",
]

DOMAINS = [
    ("Wikipedia", "https://wikipedia.com"),
    ("The New York Times", "https://nytimes.com"),
    ("Encyclopedia Britannica", "https://britannica.com"), 
#    ("a casual bodybuilding forum", "https://bodybuilding.com"),
    ("Reddit", "https://reddit.com"),
    ("a 4chan greentext with an irreverent punchline", "https://4chan.com"),
    ("a mediocre, semi-fictional short story", "https://fanfiction.net"),
#    ("Twitter", "https://twitter.com"),
    ("Reuters", 'https://reuters.com/'),
    ("an unhinged, rambling, conspiratorial manifesto", "unknown"),
    # ("Source1", 'Source1'),
    # ("Source2", 'Source2'),
]

URL_TO_NAME = {
    "https://wikipedia.com": "Wikipedia",
    "https://reddit.com": "Reddit",
    "https://nytimes.com": "New York Times",
    "https://britannica.com": "Encyclopedia Britannica",
    "https://4chan.com": "4chan",
    "https://fanfiction.net": "Fan fiction",
    "https://twitter.com": "Twitter",
    'https://reuters.com/': "Reuters",
    "unknown": "Unknown",
    "Source1": "Source1",
    "Source2": "Source2"
}

TRUSTWORTHY = set([
    "https://nytimes.com",
    "https://wikipedia.com",
    "https://britannica.com",
    'https://reuters.com/',
    'Source1'
])

UNTRUSTWORTHY = set([
    "https://reddit.com",
    "https://4chan.com",
    "https://fanfiction.net",
    "unknown",
    'Source2'
])
