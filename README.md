# S-Shape Explorer — Vietnam Travel Recommender

AI-powered Vietnam tourism chatbot that recommends real destinations based on user preferences and free-text queries.

---

## Project Structure

```
S-shape-explorer_DSAI-project/
│
├── app/                          ← FastAPI backend
│   ├── main.py                   ← Entry point, API endpoints
│   ├── models.py                 ← Pydantic schemas (Place, RecommendRequest, …)
│   ├── data_loader.py            ← Loads CSV + XLSX into Place objects
│   ├── recommender.py            ← TF-IDF similarity retrieval (uses pkl files)
│   ├── ranking.py                ← Multi-signal scorer + explanation generator
│   ├── nlp_parser.py             ← Location extractor for free-text queries
│   ├── __init__.py
│   └── static/
│       └── index.html            ← Main chat UI (served at /)
│
│
├── Full_Translated_DataSet_V2.xlsx   ← 315 places: ratings, image URLs, keywords
├── Vietnam_Tourism_Final_8Labels.csv ← 315 places: descriptions + 8 category labels
├── tfidf_model.pkl               ← Fitted TF-IDF vectorizer (315 descriptions)
├── tfidf_matrix (1).pkl          ← Pre-computed TF-IDF matrix (315 x 1000)
│
├── requirements.txt
└── .gitignore
```

---

## Dataset

| File                                | Rows | Key columns                                                                                           |
| ----------------------------------- | ---- | ----------------------------------------------------------------------------------------------------- |
| `Vietnam_Tourism_Final_8Labels.csv` | 315  | Place_Name, Description, Location, Adventure, Relax, Rural, Urban, Mountain, Historical, Food, Nature |
| `Full_Translated_DataSet_V2.xlsx`   | 315  | ID, Place_Name, Location, Description, Rating, Image_URL, Keywords                                    |

Both files are row-aligned (row 0 = same place in both).

---

## API Endpoints

| Method | Path          | Description                     |
| ------ | ------------- | ------------------------------- |
| `GET`  | `/`           | Main travel UI                  |
| `GET`  | `/health`     | Readiness probe                 |
| `POST` | `/recommend`  | Get destination recommendations |
| `GET`  | `/place/{id}` | Fetch a single place by ID      |
| `POST` | `/feedback`   | Submit like/dislike for a place |

### POST /recommend

**Request:**

```json
{
  "query": "I want a relaxing beach in Khanh Hoa",
  "preferences": ["Relax", "Nature"],
  "location": "Khanh Hoa",
  "top_k": 5
}
```

**Response:**

```json
{
  "query": "I want a relaxing beach in Khanh Hoa",
  "selected_preferences": ["Relax", "Nature"],
  "recommendations": [
    {
      "place": {
        "id": "place_001",
        "name": "Cam Ranh Long Beach",
        "location": "Khanh Hoa",
        "description": "Pristine beach with natural beauty, ideal for relaxing and swimming",
        "rating": 4.8,
        "keywords": ["sea", "play", "take photos", "relax"],
        "adventure": 0,
        "relax": 1,
        "rural": 0,
        "urban": 0,
        "mountain": 0,
        "historical": 0,
        "food": 0,
        "nature": 0
      },
      "score": 0.977,
      "matched_labels": ["Relax"],
      "explanation": "Based on your preference for Relax and Nature, I recommend Cam Ranh Long Beach in Khanh Hoa because it matches your Relax interest: Pristine beach with natural beauty, ideal for relaxing and swimming. (Rating: 4.8/5)."
    }
  ]
}
```

Valid preference values: `Adventure`, `Relax`, `Rural`, `Urban`, `Mountain`, `Historical`, `Food`, `Nature`

---

## Recommendation Logic

**Scoring formula** (weights adjust based on what the user provides):

| Case                | TF-IDF | Label match | Rating |
| ------------------- | ------ | ----------- | ------ |
| Query + Preferences | 35%    | 45%         | 20%    |
| Preferences only    | 0%     | 60%         | 40%    |
| Query only          | 65%    | 0%          | 35%    |
| Empty (fallback)    | 50%    | 0%          | 50%    |

**Location boost:**

- Exact province match: x1.50
- Partial match: x1.25
- Different location: x0.70

---

## How to Run

```bash
# 1. Install dependencies (first time only)
pip install -r requirements.txt

# 2. Start the server
uvicorn app.main:app --reload

python -m uvicorn app.main:app --reload
# 3. Open in browser
# http://127.0.0.1:8000          <- Main UI
# http://127.0.0.1:8000/docs     <- Interactive API docs
```

---

## Tech Stack

- **Backend:** FastAPI + Uvicorn
- **Recommendation:** scikit-learn TF-IDF, cosine similarity, 8-label matching
- **Data:** pandas (CSV + XLSX loading)
- **Frontend:** Vanilla HTML/CSS/JS (no framework)
