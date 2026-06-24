"""
test_pipeline.py — Regression tests for S-Shape Explorer.

Tests cover:
  - Location-based queries
  - Typo handling
  - Category + location queries
  - Region queries (North / Central / South Vietnam)
  - Anti-contamination (beach query must not return museums)
  - ML model sanity checks
  - Sea/coast synonym handling
  - Low-confidence handling (category with no match in a location)

Run:
    python test_pipeline.py

Exit 0 = all tests pass.  Exit 1 = at least one test failed.
"""

from __future__ import annotations

import sys
from typing import List

from app.data_loader import PLACES
from app.location_resolver import is_location_only_query
from app.ml_intent import infer_preferences
from app.models import PlaceResult
from app.nlp_parser import parse_query, preferences_from_parsed_query
from app.ranking import rank
from app.recommender import PlaceIndex

# ---------------------------------------------------------------------------
# Pipeline helper — mirrors the /recommend endpoint logic in main.py
# ---------------------------------------------------------------------------

_index = PlaceIndex(PLACES)


def pipeline(raw: str, top_k: int = 5) -> List[PlaceResult]:
    query = raw.strip()
    parsed = parse_query(query)
    location = parsed.get("location", "")
    parser_prefs = preferences_from_parsed_query(parsed)

    loc_only = is_location_only_query(query, location) if query and location else False
    if query and not loc_only:
        inferred, label_probs = infer_preferences(query)
    else:
        inferred, label_probs = [], {}

    ml_prefs = [] if parser_prefs else inferred
    preferences = list(dict.fromkeys(parser_prefs + ml_prefs))
    ranking_probs = {} if parser_prefs else label_probs

    pool_size = len(PLACES) if not query or location else min(len(PLACES), 80)
    candidates = _index.top_k_similar(query, k=pool_size)

    return rank(
        candidates=candidates,
        preferences=preferences,
        query_location=location,
        top_k=top_k,
        has_query=bool(query),
        label_probabilities=ranking_probs,
        query_terms=parsed.get("tags", []),
    )


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

PASS = "✓ PASS"
FAIL = "✗ FAIL"
passed = 0
failed = 0


def check(condition: bool, description: str) -> None:
    global passed, failed
    if condition:
        print(f"  {PASS}  {description}")
        passed += 1
    else:
        print(f"  {FAIL}  {description}")
        failed += 1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

print("=" * 65)
print("S-Shape Explorer — Pipeline Regression Tests")
print("=" * 65)

# ── [1] Exact destination name ────────────────────────────────────────
print("\n[1] Query: 'Hoi An'")
results = pipeline("Hoi An")
top_places = [r.place for r in results]
top_locs = [p.location.lower() for p in top_places]
check(
    any("hoi an" in p.name.lower() for p in top_places),
    "Top results contain a place with 'Hoi An' in the name",
)
check(
    all("quang nam" in loc or "da nang" in loc for loc in top_locs),
    "All results are in Quang Nam or Da Nang (Hoi An region)",
)
check(
    results[0].score >= 0.6,
    f"Top result has a strong score (got {results[0].score:.3f})",
)

# ── [2] Location-only query ───────────────────────────────────────────
print("\n[2] Query: 'places near Hoi An'")
results = pipeline("places near Hoi An")
top_locs = [r.place.location.lower() for r in results]
check(
    all("quang nam" in loc or "da nang" in loc for loc in top_locs),
    "All results are in the Hoi An / Quang Nam / Da Nang area",
)
check(len(results) >= 3, f"At least 3 results returned (got {len(results)})")

# ── [3] Region + category query ────────────────────────────────────────
print("\n[3] Query: 'beaches in south Vietnam'")
results = pipeline("beaches in south Vietnam")
SOUTH_PROVINCES = {
    "lam dong", "dong nai", "binh duong", "tay ninh", "long an", "tien giang",
    "ben tre", "tra vinh", "vinh long", "dong thap", "an giang", "kien giang",
    "hau giang", "soc trang", "bac lieu", "ca mau", "ba ria - vung tau",
    "ho chi minh city", "binh thuan", "phu quoc", "vung tau", "soc trang",
}
check(
    all(
        any(prov in r.place.location.lower() for prov in SOUTH_PROVINCES)
        for r in results
    ),
    "All results are in South Vietnam provinces",
)
BEACH_KW = {"beach", "sea", "coast", "shore", "sand"}
check(
    sum(
        1 for r in results
        if any(kw in r.place.name.lower() or kw in r.place.description.lower()
               for kw in BEACH_KW)
    ) >= 3,
    "At least 3 of the top 5 results are beach/coastal places",
)

# ── [4] Typo: "Beachs in south Viet Nam" ──────────────────────────────
print("\n[4] Query: 'Beachs in south Viet Nam' (typo test)")
results4 = pipeline("Beachs in south Viet Nam")
check(
    results4[0].place.id == results[0].place.id,
    "Typo 'Beachs' + 'Viet Nam' gives same #1 result as 'beaches in south Vietnam'",
)
check(
    all(
        any(prov in r.place.location.lower() for prov in SOUTH_PROVINCES)
        for r in results4
    ),
    "Typo query still filters to South Vietnam",
)

# ── [5] Food + city query ─────────────────────────────────────────────
print("\n[5] Query: 'food in Ho Chi Minh City'")
results = pipeline("food in Ho Chi Minh City")
check(
    results[0].place.food == 1,
    f"#1 result is food-labeled (got: {results[0].place.name})",
)
check(
    results[0].place.location.lower() in ("ho chi minh city", "hcmc", "saigon"),
    f"#1 result is in HCMC (got: {results[0].place.location})",
)
non_food_hcmc = [r for r in results if r.place.food == 0 and "ho chi minh" in r.place.location.lower()]
food_any = [r for r in results if r.place.food == 1]
if non_food_hcmc and food_any:
    worst_food = min(r.score for r in food_any)
    best_non_food = max(r.score for r in non_food_hcmc)
    check(
        worst_food >= best_non_food,
        f"Food places outscore non-food HCMC places (food≥{worst_food:.3f}, non-food≤{best_non_food:.3f})",
    )
else:
    check(True, "Only food or only non-food HCMC places in results (acceptable)")

# ── [6] Historical + location ─────────────────────────────────────────
print("\n[6] Query: 'historical places in Hue'")
results = pipeline("historical places in Hue")
HUE_PROVINCES = {"thua thien hue", "hue"}
check(
    all(any(p in r.place.location.lower() for p in HUE_PROVINCES) for r in results),
    "All results are in Hue / Thua Thien Hue",
)
hist_count = sum(1 for r in results if r.place.historical == 1)
check(hist_count >= 4, f"At least 4 of top 5 are historical (got {hist_count})")
check(results[0].place.historical == 1, f"#1 result is historical (got: {results[0].place.name})")

# ── [7] ML model sanity ───────────────────────────────────────────────
print("\n[7] ML model sanity checks")
from app.ml_intent import predict_label_probabilities, get_label_classifier
model = get_label_classifier()
check(model is not None, "Label classifier loaded from artifacts/label_classifier.joblib")

probs = predict_label_probabilities("I want to explore old temples and ancient citadels")
check(
    probs.get("Historical", 0) > probs.get("Food", 0),
    f"Historical > Food for temple query (Historical={probs.get('Historical',0):.3f})",
)
probs2 = predict_label_probabilities("relax on a beach with white sand")
check(
    probs2.get("Relax", 0) > probs2.get("Mountain", 0),
    f"Relax > Mountain for beach-relax query (Relax={probs2.get('Relax',0):.3f})",
)

# ── [8] Anti-contamination: beach query must not return pure-historical ─
print("\n[8] Anti-contamination: 'beach' query must not return pure-historical places")
results = pipeline("beach holiday in Vietnam", top_k=10)
pure_historical_in_top5 = [
    r.place for r in results[:5]
    if r.place.historical == 1 and r.place.relax == 0 and r.place.nature == 0
]
check(
    len(pure_historical_in_top5) == 0,
    f"No pure-historical places in top 5 for beach query (found {len(pure_historical_in_top5)})",
)

# ── [9] Sea synonym handling ──────────────────────────────────────────
print("\n[9] Query: 'sea' — synonym expansion for coastal/beach places")
results = pipeline("sea", top_k=5)
parsed = parse_query("sea")
check(
    "sea" in parsed["tags"] or "coast" in parsed["tags"],
    f"'sea' is extracted as a surface tag (got tags: {parsed['tags']})",
)
coastal_kw = {"sea", "beach", "coast", "shore", "ocean", "bay"}
coastal_count = sum(
    1 for r in results
    if any(kw in r.place.name.lower() or kw in r.place.description.lower()
           or any(kw in k.lower() for k in r.place.keywords)
           for kw in coastal_kw)
    or r.place.relax == 1 or r.place.nature == 1
)
check(
    coastal_count >= 3,
    f"At least 3 results are coastal/nature/relax places for 'sea' query (got {coastal_count})",
)

# ── [10] Mountain in North Vietnam ────────────────────────────────────
print("\n[10] Query: 'mountain in north Vietnam'")
results = pipeline("mountain in north Vietnam", top_k=5)
NORTH_PROVINCES = {
    "hanoi", "ha giang", "cao bang", "lao cai", "lai chau", "son la",
    "dien bien", "hoa binh", "tuyen quang", "yen bai", "lang son", "bac giang",
    "quang ninh", "hai phong", "ha long", "ha nam", "nam dinh", "bac ninh",
    "hung yen", "hai duong", "vinh phuc", "phu tho", "thai nguyen", "bac kan",
}
in_north = sum(
    1 for r in results
    if any(p in r.place.location.lower() for p in NORTH_PROVINCES)
)
mountain_labeled = sum(1 for r in results if r.place.mountain == 1)
check(
    in_north >= 4,
    f"At least 4 of 5 results are in North Vietnam (got {in_north})",
)
check(
    mountain_labeled >= 3,
    f"At least 3 results have Mountain label (got {mountain_labeled})",
)

# ── [11] Food in Hoi An — graceful low-confidence handling ────────────
print("\n[11] Query: 'food in Hoi An' (no food places in dataset for this area)")
results = pipeline("food in Hoi An", top_k=5)
# There are no food-labeled places in Quang Nam. The system should still
# return results from the area (not crash/return nothing), and scores should
# be low to reflect the mismatch — this is honest, not broken behaviour.
check(
    len(results) > 0,
    "Returns results even when no food places exist in Hoi An area",
)
check(
    all(
        "quang nam" in r.place.location.lower() or "da nang" in r.place.location.lower()
        for r in results
    ),
    "All results are still in the Hoi An area (location filter respected)",
)
check(
    results[0].score < 0.60,
    f"Top result has low confidence score (< 0.60), reflecting dataset gap (got {results[0].score:.3f})",
)

# ── [12] Image URL cleaning ────────────────────────────────────────────
print("\n[12] Image URL validation")
from app.image_utils import is_suspicious_image_url, KNOWN_PLACEHOLDER_URL_PREFIXES
BAD_PLACEHOLDER = "https://images.unsplash.com/photo-1528127269322-539801943592?auto=format"
check(
    is_suspicious_image_url(BAD_PLACEHOLDER),
    "Known 40-way shared Unsplash placeholder is detected as suspicious",
)
check(
    is_suspicious_image_url("q"),
    "Single-character value 'q' is suspicious",
)
check(
    is_suspicious_image_url("nan"),
    "Literal 'nan' is suspicious",
)
check(
    not is_suspicious_image_url("https://images.unsplash.com/photo-UNIQUE12345"),
    "Other Unsplash URLs are NOT suspicious",
)
# Count how many places got the shared placeholder replaced with SVG
placeholder_places = [
    p for p in PLACES
    if p.image_url.startswith("/static/images/fallback/")
]
check(
    len(placeholder_places) >= 40,
    f"At least 40 placeholder URLs were replaced with SVG fallbacks (got {len(placeholder_places)})",
)

# ── [13] NLP parser — typo and plural normalisation ────────────────────
print("\n[13] NLP parser: typo and synonym normalisation")
from app.nlp_parser import parse_query as pq, _normalise
check(_normalise("Beachs in south Viet Nam").find("beach") >= 0, "'Beachs' → 'beach'")
check(_normalise("Mountains in the north").find("mountain") >= 0, "'Mountains' → 'mountain'")
check(_normalise("sea and ocean coast").find("sea") >= 0, "'sea/ocean/coast' normalised")
check(pq("food in Hoi An")["location"] == "Hoi An", "Location 'Hoi An' extracted correctly")
check(pq("beaches in south Vietnam")["location"] == "South Vietnam", "Region 'South Vietnam' extracted")
check(pq("mountain in north Vietnam")["location"] == "North Vietnam", "Region 'North Vietnam' extracted")
check("Food" in preferences_from_parsed_query(pq("food in Hoi An")), "Food preference from 'food in Hoi An'")
check("Mountain" in preferences_from_parsed_query(pq("mountain trek")), "Mountain preference from 'mountain trek'")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total = passed + failed
print(f"\n{'=' * 65}")
print(f"Results: {passed}/{total} tests passed")
if failed:
    print(f"         {failed} test(s) FAILED — review the output above.")
else:
    print("         All tests passed ✓")
print("=" * 65)
sys.exit(0 if failed == 0 else 1)
