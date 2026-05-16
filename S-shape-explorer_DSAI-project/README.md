# 🗺️ Place Recommendation Engine

A production-ready Python backend for NLP-powered place recommendations using sentence-transformer embeddings, cosine similarity, and configurable multi-signal ranking.

---

## Architecture

```
POST /recommend
      │
      ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐    ┌──────────┐
│ nlp_parser  │───▶│ recommender  │───▶│    ranking       │───▶│ FastAPI  │
│ (regex NLP) │    │ (MiniLM-L6)  │    │ (weighted score) │    │ response │
└─────────────┘    └──────────────┘    └──────────────────┘    └──────────┘
      │                    │                    ▲
      ▼                    ▼                    │
  ParsedQuery        cosine sim on         feedback prefs
  { category,        100 place             (per-user tag
    mood, budget,    embeddings            weights)
    purpose, ... }
```

## File Structure

```
place_recommender/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app, endpoints
│   ├── models.py         # Pydantic schemas
│   ├── nlp_parser.py     # Regex-based NLP extractor
│   ├── recommender.py    # Embedding + similarity engine
│   ├── ranking.py        # Multi-signal weighted ranker
│   ├── data_loader.py    # JSON dataset loader
│   └── places_data.json  # 100 mock places (HCMC)
├── requirements.txt
├── sample_requests.http
└── README.md
```

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies (downloads ~90MB MiniLM model on first run)
pip install -r requirements.txt

# 3. Run the server
uvicorn app.main:app --reload --port 8000

# 4. Open interactive API docs
open http://localhost:8000/docs
```

---

## Endpoints

### `POST /recommend`
NLP-parse the query, embed it, retrieve candidates, and re-rank.

**Body:**
```json
{
  "query": "I want a quiet cafe in Ho Chi Minh City for studying, cheap price.",
  "user_id": "user_42",
  "top_k": 5,
  "weights": {
    "semantic": 0.50,
    "rating": 0.20,
    "popularity": 0.20,
    "distance": 0.10
  }
}
```

### `GET /place/{id}`
Fetch full place details by ID (e.g. `place_001`).

### `POST /feedback`
Record a like/dislike to update the user's preference vector.

```json
{ "user_id": "user_42", "place_id": "place_007", "liked": true }
```

---

## Ranking Signals

| Signal | Description | Default weight |
|--------|-------------|----------------|
| `semantic` | Cosine similarity between query and place embeddings | 0.50 |
| `rating` | Normalised 1–5 star rating | 0.20 |
| `popularity` | Normalised 1–100 popularity score | 0.20 |
| `distance` | Proximity to user GPS (if provided) | 0.10 |

Plus:
- **Budget penalty** (×0.60) if the place price level is incompatible with the parsed budget
- **Category penalty** (×0.80) for category mismatches (soft, not hard filter)
- **Preference boost** (±0.15) derived from cumulative liked/disliked tags

---

## NLP Parser

Extracts these fields using regex keyword matching:

| Field | Example values |
|-------|---------------|
| `category` | `cafe`, `restaurant`, `bar`, `park`, `coworking` |
| `mood` | `["quiet", "romantic"]` |
| `budget` | `cheap` / `moderate` / `expensive` |
| `purpose` | `["studying", "dining"]` |
| `location` | `Ho Chi Minh City` |
| `tags` | `["wifi", "rooftop"]` |

---

## Extending to Production

- **FAISS index**: Replace `cosine_similarity` in `recommender.py` with `faiss.IndexFlatIP` for 10k+ places
- **Persistent feedback**: Swap `_user_prefs` dict in `main.py` with Redis or PostgreSQL
- **Authentication**: Add `Depends(get_current_user)` to protected endpoints
- **User coordinates**: Pass `user_lat`/`user_lon` from client to enable distance ranking
- **Async embeddings**: Move embedding computation to a background task or pre-warm on startup
