"""
nlp_parser.py — Lightweight regex-based NLP extractor.

Converts a free-text user query into a structured dict:
  { category, mood, budget, purpose, location, tags }

Tuned for the Vietnam tourism dataset with 8 preference labels:
  Adventure, Relax, Rural, Urban, Mountain, Historical, Food, Nature
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Category keyword map
# Maps any recognised keyword to one of: nature, outdoor, historical,
# restaurant, attraction, relaxing.  These internal categories are later
# converted to the 8 label names by PARSER_LABEL_MAP.
# ---------------------------------------------------------------------------

CATEGORY_MAP: dict[str, str] = {
    # Nature / water
    "nature": "nature", "beach": "nature", "sea": "nature",
    "ocean": "nature", "coast": "nature", "coastal": "nature",
    "waterfall": "nature", "forest": "nature", "lake": "nature",
    "river": "nature", "island": "nature", "cave": "nature",
    "national park": "nature", "park": "nature", "shore": "nature",
    "bay": "nature",
    # Mountain / trekking
    "mountain": "outdoor", "hill": "outdoor", "highland": "outdoor",
    "peak": "outdoor", "pass": "outdoor",
    "trek": "outdoor", "trekking": "outdoor",
    "hike": "outdoor", "hiking": "outdoor",
    "climbing": "outdoor",
    # Adventure / sport
    "adventure": "outdoor", "sport": "outdoor", "active": "outdoor",
    "outdoor": "outdoor", "amusement": "outdoor", "zip line": "outdoor",
    # Historical / cultural
    "historical": "historical", "history": "historical", "temple": "historical",
    "pagoda": "historical", "citadel": "historical", "relic": "historical",
    "heritage": "historical", "ancient": "historical", "museum": "historical",
    "monument": "historical", "church": "historical", "cultural": "historical",
    "culture": "historical", "traditional": "historical",
    # Food
    "food": "restaurant", "eat": "restaurant", "restaurant": "restaurant",
    "market": "restaurant", "night market": "restaurant", "cuisine": "restaurant",
    "street food": "restaurant", "specialty": "restaurant", "dining": "restaurant",
    "culinary": "restaurant",
    # Urban / city
    "city": "attraction", "urban": "attraction", "town": "attraction",
    "shopping": "attraction", "entertainment": "attraction",
    # Relaxation
    "relax": "relaxing", "resort": "relaxing", "spa": "relaxing",
    "rest": "relaxing", "unwind": "relaxing", "chill": "relaxing",
    # Rural / village
    "village": "nature", "rural": "nature", "countryside": "nature",
    "ethnic": "nature", "tribe": "nature",
}

MOOD_KEYWORDS: dict[str, list[str]] = {
    "relaxing":    ["relax", "relaxing", "peaceful", "calm", "quiet", "tranquil",
                    "serene", "chill", "unwind", "rest", "slow"],
    "adventurous": ["adventure", "adventurous", "exciting", "thrill", "active",
                    "extreme", "challenge", "sport", "trek", "hike", "climb"],
    "romantic":    ["romantic", "couple", "honeymoon", "intimate", "date", "love"],
    "cultural":    ["cultural", "history", "historical", "heritage", "traditional",
                    "ancient", "learn", "educational"],
    "social":      ["social", "group", "family", "friends", "community", "festival"],
    "scenic":      ["scenic", "beautiful", "stunning", "view", "landscape",
                    "photography", "instagram", "picturesque"],
    "nature":      ["nature", "natural", "green", "forest", "mountain", "outdoor",
                    "wildlife", "fresh air", "eco"],
    "spiritual":   ["spiritual", "temple", "pagoda", "pray", "meditation", "sacred"],
}

BUDGET_MAP: dict[str, list[str]] = {
    "cheap":     ["cheap", "affordable", "budget", "inexpensive", "low cost",
                  "low-cost", "economical", "free", "value", "pocket friendly"],
    "moderate":  ["moderate", "mid-range", "mid range", "reasonable", "fair price",
                  "not too expensive"],
    "expensive": ["expensive", "luxury", "premium", "upscale", "high-end", "splurge"],
}

PURPOSE_KEYWORDS: dict[str, list[str]] = {
    "sightseeing": ["sightseeing", "explore", "visit", "tour", "see", "discover"],
    "photography": ["photo", "photograph", "photography", "instagram", "picture",
                    "selfie", "shoot"],
    "trekking":    ["trek", "trekking", "hike", "hiking", "trail", "walk"],
    "swimming":    ["swim", "swimming", "beach", "snorkel", "dive", "water"],
    "learning":    ["learn", "study", "educational", "history", "museum",
                    "understand", "knowledge"],
    "eating":      ["eat", "food", "dining", "try food", "taste", "cuisine",
                    "local food", "street food"],
    "relaxing":    ["relax", "rest", "unwind", "chill", "holiday", "vacation",
                    "getaway", "retreat"],
    "camping":     ["camp", "camping", "overnight", "tent", "glamping"],
    "family":      ["family", "kids", "children", "parents", "family trip"],
}

# Vietnam provinces and major city/destination names (lowercase, no diacritics).
VIETNAM_LOCATIONS: list[str] = [
    "hanoi", "ho chi minh city", "hcmc", "saigon", "da nang", "hoi an",
    "hue", "nha trang", "da lat", "dalat", "phu quoc", "ha long", "halong",
    "sapa", "sa pa", "ninh binh", "mekong delta", "can tho", "mui ne",
    "quy nhon", "vung tau", "con dao", "ha giang", "cao bang", "lao cai",
    "lai chau", "son la", "dien bien", "hoa binh", "tuyen quang", "yen bai",
    "lang son", "bac giang", "quang ninh", "hai phong", "ha nam", "nam dinh",
    "ninh thuan", "binh thuan", "khanh hoa", "quang nam", "quang binh",
    "quang tri", "thua thien hue", "binh dinh", "phu yen", "gia lai",
    "kon tum", "dak lak", "dak nong", "lam dong", "dong nai", "binh duong",
    "tay ninh", "long an", "tien giang", "ben tre", "tra vinh", "vinh long",
    "dong thap", "an giang", "kien giang", "hau giang", "soc trang",
    "bac lieu", "ca mau", "bac ninh", "hung yen", "hai duong", "vinh phuc",
    "phu tho", "thai nguyen", "bac kan", "ha tinh", "nghe an", "thanh hoa",
    "quang ngai", "binh phuoc",
]

# Regional grouping — used to expand "North Vietnam" → list of provinces.
REGION_MAP: dict[str, list[str]] = {
    "north": [
        "hanoi", "ha giang", "cao bang", "lao cai", "sapa", "sa pa",
        "lai chau", "son la", "dien bien", "hoa binh", "tuyen quang", "yen bai",
        "lang son", "bac giang", "quang ninh", "hai phong", "ha long", "halong",
        "ha nam", "nam dinh", "bac ninh", "hung yen", "hai duong", "vinh phuc",
        "phu tho", "thai nguyen", "bac kan",
    ],
    "central": [
        "ha tinh", "nghe an", "thanh hoa", "quang tri", "quang binh",
        "thua thien hue", "hue", "da nang", "quang nam", "quang ngai",
        "binh dinh", "quy nhon", "phu yen", "khanh hoa", "nha trang",
        "ninh thuan", "ninh binh",
    ],
    "south": [
        "lam dong", "da lat", "dalat", "dong nai", "binh duong", "tay ninh",
        "long an", "tien giang", "ben tre", "tra vinh", "vinh long",
        "dong thap", "an giang", "kien giang", "hau giang", "soc trang",
        "bac lieu", "ca mau", "vung tau", "con dao", "ho chi minh city", "hcmc",
        "saigon", "binh thuan", "mui ne", "phu quoc",
    ],
}

LOCATION_PATTERNS: list[str] = [
    r"\bin\s+(ho chi minh city)",
    r"\bin\s+([\w\s]{3,30})",
    r"\bnear\s+([\w\s]{3,20})",
    r"\bat\s+([\w\s]{3,20})",
]

# Surface-level tags extracted from the query.
# These are used by ranking.py to boost places that actually mention the term.
GENERIC_TAGS: list[str] = [
    "beach", "sea", "coast", "mountain", "waterfall", "cave", "lake",
    "river", "island", "forest", "temple", "pagoda", "market", "village",
    "hiking", "trekking", "swimming", "camping", "festival",
    "night market", "street food", "hot spring",
    "rice terrace", "rice field", "boat", "cruise",
    "sunrise", "sunset", "view", "landscape",
    # Food intent tags
    "food", "restaurant", "eat", "dining", "cuisine",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """
    Lowercase, strip diacritics, and normalise common plural/typo variants.

    Why we normalise:
    - "Viet Nam" and "Vietnam" are the same; we collapse them so neither
      dominates semantic retrieval (the whole dataset is about Vietnam).
    - Plurals like "beaches", "beachs" (typo), "mountains" all map to
      their singular form so keyword matching works correctly.
    - "sea"/"seas"/"ocean"/"coast"/"coastal" are treated as beach synonyms.
    """
    text = text.casefold().strip()
    # Strip diacritics (e.g. Vietnamese tone marks when typed without IME)
    text = "".join(
        char for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )
    # Normalise country name variants
    text = re.sub(r"\bviet\s+nam\b", "vietnam", text)
    # Plural / typo normalisation for common nouns
    text = re.sub(r"\bbeach(?:es|s)?\b", "beach", text)
    text = re.sub(r"\bmountains?\b",     "mountain", text)
    text = re.sub(r"\bislands?\b",       "island", text)
    text = re.sub(r"\bwaterfalls?\b",    "waterfall", text)
    text = re.sub(r"\bmarkets?\b",       "market", text)
    text = re.sub(r"\btemples?\b",       "temple", text)
    text = re.sub(r"\bpagodas?\b",       "pagoda", text)
    text = re.sub(r"\bseas?\b",          "sea", text)
    text = re.sub(r"\boceans?\b",        "sea", text)
    text = re.sub(r"\bcoasts?\b",        "coast", text)
    return re.sub(r"\s+", " ", text).strip()


def _match_keywords(text: str, keyword_map: dict[str, list[str]]) -> list[str]:
    """Return every canonical key whose keyword list has a match in text."""
    matched = []
    for canonical, keywords in keyword_map.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text):
                matched.append(canonical)
                break
    return matched


def _extract_location(text: str) -> str:
    """
    Extract a location hint from the query text.

    Priority order:
    1. Specific province / city name  ("da nang", "nha trang", …)
    2. Regional keyword               ("northern", "south vietnam", …)
    3. Generic positional pattern     ("in X", "near X", "at X")
    """
    # 1. Specific place / province names
    for loc in VIETNAM_LOCATIONS:
        if re.search(r"\b" + re.escape(loc) + r"\b", text):
            return loc.title()

    # 2. Regional keywords — checked before generic "in X" so "in Northern
    #    Vietnam" maps to a region, not a string starting with "Northern".
    regional_keywords: dict[str, list[str]] = {
        "north":   [r"\bnorth(?:ern)?\s+(?:vietnam|viet)\b", r"\bnorth(?:ern)?\b"],
        "central": [r"\bcentral\s+(?:vietnam|viet)\b",        r"\bcentral\b"],
        "south":   [r"\bsouth(?:ern)?\s+(?:vietnam|viet)\b",  r"\bsouth(?:ern)?\b"],
        "mekong":  [r"\bmekong(?:\s+(?:delta|region))?\b"],
    }
    for region, patterns in regional_keywords.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "Mekong Delta" if region == "mekong" else region.capitalize() + " Vietnam"

    # 3. Generic positional patterns
    for pattern in LOCATION_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            captured = m.group(1).strip()
            # "in Vietnam" is not a useful constraint — the whole dataset is Vietnam.
            if captured.lower() in {"vietnam", "viet nam", "viet", "nam", "vn"}:
                return ""
            return captured.title()

    return ""


# Maps internal category/mood/purpose keys to the 8 label names.
PARSER_LABEL_MAP: dict[str, list[str]] = {
    # Internal category keys
    "nature":       ["Nature"],
    "outdoor":      ["Adventure", "Mountain"],
    "historical":   ["Historical"],
    "restaurant":   ["Food"],
    "attraction":   ["Urban"],
    "relaxing":     ["Relax"],
    # Mood keys
    "adventurous":  ["Adventure"],
    "cultural":     ["Historical"],
    "scenic":       ["Nature"],
    "spiritual":    ["Historical"],
    # Purpose keys
    "eating":       ["Food"],
    "trekking":     ["Adventure", "Mountain"],
    "swimming":     ["Relax", "Nature"],
    "learning":     ["Historical"],
    "camping":      ["Adventure", "Nature"],
    "family":       ["Relax"],
    # Tag keys
    "beach":        ["Relax", "Nature"],
    "sea":          ["Relax", "Nature"],
    "coast":        ["Relax", "Nature"],
    "mountain":     ["Mountain", "Adventure"],
    "waterfall":    ["Nature", "Adventure"],
    "cave":         ["Nature", "Adventure"],
    "lake":         ["Nature", "Relax"],
    "river":        ["Nature", "Relax"],
    "island":       ["Nature", "Relax"],
    "forest":       ["Nature"],
    "temple":       ["Historical"],
    "pagoda":       ["Historical"],
    "market":       ["Food", "Urban"],
    "night market": ["Food", "Urban"],
    "street food":  ["Food"],
    "village":      ["Rural"],
    "rice terrace": ["Rural", "Mountain", "Nature"],
    "rice field":   ["Rural", "Nature"],
    "food":         ["Food"],
    "eat":          ["Food"],
    "dining":       ["Food"],
    "cuisine":      ["Food"],
    "restaurant":   ["Food"],
}


def preferences_from_parsed_query(parsed_query: dict) -> list[str]:
    """
    Convert deterministic parser signals into the 8 model label names.

    This prevents clear keyword queries (e.g. "food in Hoi An") from being
    overruled by the statistical ML classifier when the dataset is small.
    """
    labels: list[str] = []

    def _add(key: str) -> None:
        for label in PARSER_LABEL_MAP.get(key, []):
            if label not in labels:
                labels.append(label)

    _add(parsed_query["category"])
    for field in ("mood", "purpose", "tags"):
        for value in parsed_query[field]:
            _add(value)

    return labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_query(raw_query: str) -> dict:
    """
    Parse a natural-language Vietnam tourism query into structured intent.

    Returns a dict with keys: category, mood, budget, purpose, location, tags.
    """
    text = _normalise(raw_query)

    # Category — match longest phrase first to prefer "night market" over "market"
    category = ""
    for phrase, canonical in sorted(CATEGORY_MAP.items(), key=lambda x: -len(x[0])):
        if re.search(r"\b" + re.escape(phrase) + r"\b", text):
            category = canonical
            break

    mood    = _match_keywords(text, MOOD_KEYWORDS)
    budget  = ""
    for level, keywords in BUDGET_MAP.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text):
                budget = level
                break
        if budget:
            break

    purpose  = _match_keywords(text, PURPOSE_KEYWORDS)
    location = _extract_location(text)

    tags: list[str] = []
    for tag in GENERIC_TAGS:
        if re.search(r"\b" + re.escape(tag) + r"\b", text):
            tags.append(tag)

    return {
        "category": category,
        "mood":     mood,
        "budget":   budget,
        "purpose":  purpose,
        "location": location,
        "tags":     list(dict.fromkeys(tags)),  # preserve order, remove duplicates
    }
