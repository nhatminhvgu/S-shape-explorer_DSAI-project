# S-Shape Explorer — Vietnam AI Travel Recommender

A FastAPI + HTML/JS tourism recommendation app for Vietnam using TF-IDF text similarity, multi-label classification, and a multi-signal ranking formula.

---

## What the App Does

1. The user types a free-text query ("beach in south Vietnam", "trekking in Sapa", "food in Hanoi").
2. A regex-based NLP parser extracts: **location**, **category tags**, **mood**, and **purpose**.
3. A trained **multi-label classifier** (scikit-learn) predicts probabilities across 8 preference labels.
4. **TF-IDF cosine similarity** retrieves the most semantically relevant places from 315 real Vietnam destinations.
5. A **multi-signal ranking formula** combines semantic score, label match, ML confidence, dataset rating, and a location boost multiplier.
6. The top results are returned with **AI-generated explanations** and a **match confidence score**.

---

## AI / Data Science Pipeline

| Stage                | Method                                                         | File                                       |
| -------------------- | -------------------------------------------------------------- | ------------------------------------------ |
| Text preprocessing   | Regex normalisation, typo/plural correction, synonym expansion | `app/nlp_parser.py`, `app/recommender.py`  |
| Intent detection     | Rule-based keyword extraction                                  | `app/nlp_parser.py`                        |
| Label classification | Trained OneVsRest + TF-IDF classifier                          | `app/ml_intent.py`, `train_label_model.py` |
| Similarity retrieval | TF-IDF + cosine similarity                                     | `app/recommender.py`                       |
| Ranking              | Weighted multi-signal formula                                  | `app/ranking.py`                           |
| Feedback loop        | Dampened Bayesian rating update                                | `app/main.py`                              |

### Ranking Formula

```
Score = α·semantic + β·label_match + γ·ai_classifier + δ·rating
```

Weights adapt to what is available:

| Condition          | α    | β    | γ    | δ    |
| ------------------ | ---- | ---- | ---- | ---- |
| Query + prefs + ML | 0.30 | 0.30 | 0.20 | 0.20 |
| Query + prefs      | 0.35 | 0.45 | —    | 0.20 |
| Query only         | 0.65 | —    | —    | 0.35 |
| Prefs only         | —    | 0.60 | —    | 0.40 |

Location boost multipliers: ×2.00 (same city/province), ×1.30 (same region), ×0.55 (off-location penalty).

---

## Dataset

- **315 real Vietnam destinations** sourced from travel review data
- **8 binary preference labels**: Adventure, Relax, Rural, Urban, Mountain, Historical, Food, Nature
- Source files: `Vietnam_Tourism_Final_8Labels.csv` + `Full_Translated_DataSet_V2.xlsx`

### Label Distribution

| Label      | Count |
| ---------- | ----- |
| Relax      | 104   |
| Nature     | 94    |
| Historical | 72    |
| Urban      | 58    |
| Mountain   | 60    |
| Adventure  | 43    |
| Food       | 22    |
| Rural      | 26    |

> **Known limitation**: The dataset has very few Food-labeled places (22/315 = 7%). Queries like "food in Hoi An" may return low-confidence results because there are no food places recorded for that province in the dataset. The system detects this and shows a warning banner.

---

## Quick Start (Windows PowerShell)

```execute chatbot
# 1. Install dependencies (first time only)
pip install -r requirements.txt

# 2. Start the server
python -m uvicorn app.main:app --reload

# 3. Open in browser
# http://127.0.0.1:8000          <- Main UI
# http://127.0.0.1:8000/docs     <- Interactive API docs
```

````visualization

# Install visualization dependencies (first time only)
pip install matplotlib seaborn scikit-learn

# Core charts (6 PNGs) — label distribution, TF-IDF heatmap,
# cosine similarity matrix, rating histogram, scoring weights
python visualization/visualize_model.py

# Extra charts (4 PNGs) — PCA clustering, top keywords per category,
# recommendation score breakdown, system pipeline diagram
python visualization/visualize_extra.py

# Output is saved to: visualization/charts/
---

## Run Tests

```powershell
# Run all 39 regression tests
python test_pipeline.py

# Run evaluation metrics
python evaluate_recommender.py
````

---

## Project Structure

```
├── app/
│   ├── main.py            # FastAPI endpoints
│   ├── data_loader.py     # Loads CSV + XLSX dataset
│   ├── models.py          # Pydantic data models
│   ├── nlp_parser.py      # Regex-based query intent parser
│   ├── ml_intent.py       # Trained multi-label classifier
│   ├── recommender.py     # TF-IDF retrieval engine
│   ├── ranking.py         # Multi-signal ranking formula
│   ├── location_resolver.py # Location alias and region logic
│   ├── image_utils.py     # Image URL validation
│   └── static/
│       └── index.html     # Frontend UI
├── Vietnam_Tourism_Final_8Labels.csv
├── Full_Translated_DataSet_V2.xlsx
├── tfidf_model.pkl
├── tfidf_matrix (1).pkl
├── train_label_model.py   # Retrain the classifier
├── test_pipeline.py       # 39 regression tests
├── evaluate_recommender.py
└── requirements.txt
```

---

## Known Limitations

1. **Small dataset**: 315 places is enough to demonstrate the system but too few for robust generalisation. Some provinces have only 1–2 entries.
2. **Imbalanced labels**: Food (22 places) and Rural (26 places) are underrepresented. The classifier performs worse on these.
3. **Image URL quality**: ~144 of 315 image URLs come from Bing/Google thumbnail proxies or a shared placeholder. These may break without notice. Broken images fall back to local SVG icons.
4. **No real user click data**: Recommendation evaluation uses proxy relevance (label match + location match), not actual user behaviour.
5. **TF-IDF vocabulary**: Typos and very rare terms may not appear in the vocabulary. Synonym expansion and regex normalisation mitigate this.
6. **Static dataset**: Ratings and descriptions reflect the dataset snapshot, not live review data.

---

## Image URL Status

See `image_url_manual_review.csv` for the full list.

- **40 places** used the same shared Unsplash placeholder (`photo-1528127269322-539801943592`) — these now display category SVG fallbacks.
- **52 places** use Bing thumbnail proxies (may expire without notice).
- **51 places** use Google search thumbnails (high breakage risk).
- **1 place** had an invalid URL (`q`) — replaced with SVG fallback.
- The browser `onerror` handler automatically falls back to SVG icons for any URL that fails to load.

---

## Defense Notes

**"How does the recommendation work?"**

> The user's query goes through three stages: (1) a regex parser extracts location and category signals, (2) a trained multi-label scikit-learn classifier predicts probabilities across 8 labels, (3) TF-IDF cosine similarity retrieves similar places, and (4) a weighted formula combining semantic score, label match, classifier confidence, and rating ranks the final results.

**"Why TF-IDF instead of a neural model?"**

> Our dataset has only 315 entries — too few for fine-tuning a transformer model without overfitting. TF-IDF is interpretable, fast, and produces reasonable results on this scale. It is also easy to explain in a university setting.

**"How did you evaluate the system?"**

> We used proxy relevance (label match + location match) since we have no real user click data. Test cases cover typo handling, regional filtering, anti-contamination (beach queries must not return museums), and low-confidence detection. See `test_pipeline.py` and `evaluate_recommender.py`.

**"What would you improve with more time?"**

> Expand the dataset, add real user feedback collection, replace TF-IDF with a sentence-transformer for better semantic understanding, fix broken image URLs, and add a Vietnamese-language query interface.
> #   F i n a l - D S A I 
>  
>  
