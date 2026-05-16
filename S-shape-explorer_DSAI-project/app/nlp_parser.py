"""
nlp_parser.py — Lightweight regex-based NLP extractor.

Converts a free-text user query into a structured ParsedQuery JSON object:
  { category, mood, budget, purpose, location, tags }

Tuned for Vietnam tourism dataset with 8 labels:
  Adventure, Relax, Rural, Urban, Mountain, Historical, Food, Nature
"""

import re
from app.models import ParsedQuery

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

CATEGORY_MAP: dict[str, str] = {
    # Nature / outdoors
    "nature": "nature", "beach": "nature", "waterfall": "nature",
    "forest": "nature", "lake": "nature", "river": "nature",
    "island": "nature", "cave": "nature", "national park": "nature",
    "park": "nature", "sea": "nature", "coast": "nature",
    # Mountain / trekking
    "mountain": "nature", "hill": "nature", "trek": "outdoor",
    "trekking": "outdoor", "hike": "outdoor", "hiking": "outdoor",
    "climbing": "outdoor", "bouldering": "outdoor",
    # Adventure
    "adventure": "outdoor", "sport": "outdoor", "active": "outdoor",
    "outdoor": "outdoor", "amusement": "outdoor", "zip line": "outdoor",
    # Historical / cultural
    "historical": "historical", "history": "historical", "temple": "historical",
    "pagoda": "historical", "citadel": "historical", "relic": "historical",
    "heritage": "historical", "ancient": "historical", "museum": "historical",
    "monument": "historical", "church": "historical", "cultural": "historical",
    "culture": "historical",
    # Food
    "food": "restaurant", "eat": "restaurant", "restaurant": "restaurant",
    "market": "restaurant", "night market": "restaurant", "cuisine": "restaurant",
    "street food": "restaurant", "specialty": "restaurant", "dining": "restaurant",
    # Urban / city
    "city": "attraction", "urban": "attraction", "town": "attraction",
    "shopping": "attraction", "entertainment": "attraction",
    # Relaxation
    "relax": "nature", "resort": "nature", "spa": "nature",
    "rest": "nature", "unwind": "nature",
    # Rural / village
    "village": "nature", "rural": "nature", "countryside": "nature",
    "ethnic": "nature", "tribe": "nature", "traditional": "historical",
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

# Vietnam provinces and major destinations
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

LOCATION_PATTERNS: list[str] = [
    r"\bin\s+(ho chi minh city)",
    r"\bin\s+([\w\s]{3,30})",
    r"\bnear\s+([\w\s]{3,20})",
    r"\bat\s+([\w\s]{3,20})",
]

GENERIC_TAGS: list[str] = [
    "beach", "mountain", "waterfall", "cave", "lake", "river", "island",
    "forest", "temple", "pagoda", "market", "village", "hiking", "trekking",
    "swimming", "camping", "festival", "night market", "street food",
    "hot spring", "rice terrace", "rice field", "boat", "cruise",
    "sunrise", "sunset", "view", "landscape",
]

REJECTION_KEYWORDS = [
    "no", "not", "none", "rejection", "dislike", 
    "no way", "I do not like it", "something else", "reject",
    "avoid", "do not want", "don't want", "no thanks", "not interested",

]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _match_keywords(text: str, keyword_map: dict[str, list[str]]) -> list[str]:
    matched = []
    for canonical, keywords in keyword_map.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text):
                matched.append(canonical)
                break
    return matched


def _extract_location(text: str) -> str:
    for loc in VIETNAM_LOCATIONS:
        if re.search(r"\b" + re.escape(loc) + r"\b", text):
            return loc.title()
    for pattern in LOCATION_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().title()
    return ""

def detect_rejection(raw_query: str) -> bool:
    text = raw_query.lower().strip()
    for kw in REJECTION_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", text):
            return True
    return False

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_query(raw_query: str) -> ParsedQuery:
    """
    Extract structured intent from a natural-language Vietnam tourism query.
    Returns a ParsedQuery with best-effort values.
    """
    text = _normalise(raw_query)

    # --- Category -----------------------------------------------------------
    category = ""
    for phrase, canonical in sorted(CATEGORY_MAP.items(), key=lambda x: -len(x[0])):
        if re.search(r"\b" + re.escape(phrase) + r"\b", text):
            category = canonical
            break

    # --- Mood ---------------------------------------------------------------
    mood = _match_keywords(text, MOOD_KEYWORDS)

    # --- Budget -------------------------------------------------------------
    budget = ""
    for level, keywords in BUDGET_MAP.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text):
                budget = level
                break
        if budget:
            break

    # --- Purpose ------------------------------------------------------------
    purpose = _match_keywords(text, PURPOSE_KEYWORDS)

    # --- Location -----------------------------------------------------------
    location = _extract_location(text)

    # --- Tags ---------------------------------------------------------------
    tags: list[str] = []
    for tag in GENERIC_TAGS:
        if re.search(r"\b" + re.escape(tag) + r"\b", text):
            tags.append(tag)

    seen: set[str] = set()
    unique_tags = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            unique_tags.append(t)

    return ParsedQuery(
        category=category,
        mood=mood,
        budget=budget,
        purpose=purpose,
        location=location,
        tags=unique_tags,
    )
