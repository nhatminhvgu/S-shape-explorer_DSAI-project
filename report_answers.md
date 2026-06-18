# GROUP PROJECT REPORT — S-Shape Explorer Assistance

---

## Abstract
*Write a short summary of the whole project in 150–250 words.*

S-Shape Explorer is a travel recommendation assistant built for Vietnam tourism. The problem we're trying to solve is pretty simple: when people want to travel in Vietnam, they don't always know where to go, and searching through tons of lists online is kind of a pain. So we built a chatbot-style system that takes a user's text query, their travel preferences (like Adventure, Relax, Nature, etc.), and optionally a location, then returns a ranked list of recommended places.

The dataset consists of 315 real Vietnam tourism destinations stored in two files: a CSV with place names, locations, descriptions, and 8 binary preference labels, and an XLSX file with ratings, keywords, and image URLs. We pre-processed the descriptions and trained a TF-IDF model on them.

The main AI/ML method is TF-IDF (Term Frequency-Inverse Document Frequency) for semantic similarity, combined with a multi-signal ranking formula. The final score for each place is a weighted combination of TF-IDF cosine similarity (35%), preference label matching (45%), and user rating (20%), with a location boost multiplier on top.

The system successfully returns relevant, explainable recommendations. The final output is a working web application with a chatbot-style interface and a FastAPI backend that serves the recommendations in real time.

**Checklist:**
- [x] State the problem.
- [x] Mention the dataset.
- [x] Mention the AI/ML method used.
- [x] Summarize the main result.
- [x] State the final output or contribution.

---

## Introduction
*Explain the background and motivation of the project.*

Vietnam has a huge number of tourism spots — mountains, beaches, historical sites, cities, countryside — spread across a long S-shaped coastline from north to south (hence the project name). The problem is that most tourists, especially first-timers, have no idea where to start. Existing platforms like TripAdvisor or Google Maps are helpful but require a lot of manual filtering and don't really understand natural language like "I want something quiet and natural near Hue."

That's the motivation here. We wanted to build something that feels more like a conversation — you tell the system what you're looking for, and it gives you a list of places that actually match, not just based on raw keywords but on broader travel preferences.

The goal of this project is to build a working recommendation system for Vietnam tourism that takes free-text queries and preference tags as input and returns ranked, explainable destination recommendations.

This could benefit tourists planning a trip, local travel agencies wanting to automate advice, or anyone who wants to explore Vietnam without spending hours researching.

**Checklist:**
- [x] What is the problem?
- [x] Why is this problem important?
- [x] Who may benefit from solving it?
- [x] What is the goal of your project?

---

## Related Work / Literature Review
*Briefly describe existing studies, systems, datasets, or methods related to your project.*

A few things we looked at before building this:

**1. TF-IDF for Content-Based Recommendation**
TF-IDF is one of the more classic text retrieval methods and has been used in content-based filtering for a while. Burke (2002) in *Hybrid Recommender Systems: Survey and Experiments* explains how content-based methods like TF-IDF work well when you have good item descriptions but limited user interaction history. That's exactly our situation — we have descriptions for 315 places but no historical user data.

**2. Tourism Recommendation Systems**
Loh et al. (2003) in *A Tourism Recommender System Based on Collaboration and Text Analysis* show that preference-based filtering is a valid approach for tourism domains. Their work uses user profiles defined by travel categories, which is similar to our 8-label system (Adventure, Relax, Rural, Urban, Mountain, Historical, Food, Nature).

**3. ML-backed REST APIs with FastAPI and scikit-learn**
Various tutorials and the official FastAPI documentation show patterns for serving a pre-trained model via REST endpoints. We followed the same approach: pre-train the TF-IDF offline, serialize it with pickle, and load it at startup.

Where our project differs: we combine TF-IDF similarity with rule-based label matching and a location boost in a single weighted scoring formula, which makes the recommendation more practical for this specific domain.

**References:**
- Burke, R. (2002). Hybrid Recommender Systems: Survey and Experiments. *User Modeling and User-Adapted Interaction*, 12(4), 331–370.
- Loh, S., Lorenzi, F., Saldaña, R., & Licthnow, D. (2003). A Tourism Recommender System Based on Collaboration and Text Analysis. *Information Technology & Tourism*, 6(3), 157–165.
- scikit-learn Developers. (2024). Feature extraction — Text feature extraction. https://scikit-learn.org/stable/modules/feature_extraction.html

**Checklist:**
- [x] Review at least 2–3 related sources.
- [x] Explain how they are connected to your project.
- [x] Mention what your project does similarly or differently.
- [x] Include citations or references.

---

## Problem Statement
*Clearly define the task your project solves.*

**Input:** A user-provided free-text query (e.g., "I want a relaxing beach in Khanh Hoa"), a list of preference tags (e.g., ["Relax", "Nature"]), an optional location filter, and a desired number of results (top_k, between 1 and 20).

**Expected output:** A ranked list of top-k Vietnam tourism destinations, each with a relevance score, matched preference labels, and a natural-language explanation of why the place was recommended.

**AI/ML task:** Given a user query and/or preference labels, our system aims to recommend the most relevant Vietnam tourism destinations from a database of 315 real places.

**Checklist:**
- [x] State the input.
- [x] State the expected output.
- [x] Define the AI/ML task type.

**Task type:**
- [x] Recommendation System

---

## Dataset
*Describe the data used in the project.*

The dataset covers 315 real Vietnam tourism destinations, split across two files:

**Vietnam_Tourism_Final_8Labels.csv** — main dataset:
- `Place_Name`: name of the destination
- `Location`: province or city in Vietnam
- `Description`: short text description of the place
- 8 binary label columns: `Adventure`, `Relax`, `Rural`, `Urban`, `Mountain`, `Historical`, `Food`, `Nature` (each 0 or 1)

**Full_Translated_DataSet_V2.xlsx** — supplementary data:
- `Rating`: e.g., "4.8/5" (parsed to float at load time)
- `Keywords`: comma-separated descriptive tags
- `Image_URL`: link to a photo of the place

Both files have exactly 315 rows aligned by row index. There are also two pre-trained TF-IDF pickle files (`tfidf_model.pkl`, `tfidf_matrix (1).pkl`) built on the 315 descriptions with a vocabulary of approximately 1,000 terms.

There is no explicit target variable — this is a retrieval/ranking task, not a supervised classification problem.

**Limitations:** The dataset is relatively small (315 places compared to thousands of real destinations in Vietnam). Descriptions are in English (translated from Vietnamese), and the 8 labels were manually assigned, so there may be some inconsistency. There is also no user interaction history, which rules out collaborative filtering.

**Data source:**
- [x] Web scraping / API

**Checklist:**
- [x] Name or describe the dataset.
- [x] State the data source.
- [x] State the number of samples, if known.
- [x] Describe the main features/columns.
- [x] Describe the target variable, if applicable.
- [x] Mention any data limitations.

---

## Methodology
*Explain how the project was implemented.*

**Preprocessing:**
- CSV and XLSX files are loaded with pandas at server startup
- Ratings are parsed from "4.8/5" format to a float value (4.8)
- Keywords are split on commas and stripped of surrounding quotes
- All text queries are lowercased before TF-IDF transformation

**Feature Engineering:**
- Place descriptions were used to train a TF-IDF vectorizer with a vocabulary of ~1,000 terms (stored as `tfidf_model.pkl`)
- The pre-computed TF-IDF matrix for all 315 descriptions is stored as `tfidf_matrix (1).pkl`
- 8 binary preference labels serve as structured categorical features for label-based matching

**Model / Algorithm — Two-stage pipeline:**

*Stage 1 — TF-IDF Retrieval (`recommender.py`):*
The user query is transformed with the pre-fitted TF-IDF vectorizer and cosine similarity is computed against the full 315-row matrix. This returns a ranked pool of candidate places. If no text query is given, all 315 places receive a neutral score of 0.5 so the ranking stage decides.

*Stage 2 — Multi-signal Re-ranking (`ranking.py`):*
Each candidate is re-scored using a weighted formula depending on what the user provided:

| Case | TF-IDF | Label Match | Rating |
|------|--------|-------------|--------|
| Query + Preferences | 35% | 45% | 20% |
| Preferences only | 0% | 60% | 40% |
| Query only | 65% | 0% | 35% |
| Nothing provided | 50% | 0% | 50% |

A location boost multiplier is then applied: ×1.50 for exact match, ×1.25 for partial match, ×0.70 for no match.

**Why this method is suitable:**
TF-IDF is lightweight and works well for content-based retrieval when there is no user history. The 8-label system adds structured preference matching on top of unstructured text similarity. The whole pipeline is near-instant because the TF-IDF model is pre-trained and loaded from pickle once at startup.

**Validation:**
No traditional train/test split was done — this is a retrieval system with no labeled query-result ground truth. Validation was done manually by running a variety of test queries and checking that the returned results made sense.

**Tools and Libraries:**
- Python 3.x
- FastAPI (web framework + REST API)
- scikit-learn (TF-IDF vectorizer, cosine similarity)
- pandas (data loading and processing)
- Pydantic (request/response validation)
- pickle (model serialization)
- HTML / CSS / JavaScript (frontend UI)

**Checklist:**
- [x] Describe preprocessing steps.
- [x] Describe feature engineering, if any.
- [x] Describe the model(s) or algorithm(s) used.
- [x] Explain why the method is suitable.
- [x] Describe the train/test split or validation method.
- [x] Mention tools and libraries used.

---

## Experiments and Results
*Present the results of your model or system.*

Since this is a recommendation system with no labeled query-result test set, we evaluated through manual/human evaluation. We ran a set of test queries and judged whether the top-k results were relevant.

**Test results:**

| Query | Preferences | Location | Top Result | Relevant? |
|-------|-------------|----------|------------|-----------|
| "relaxing beach" | Relax, Nature | Khanh Hoa | Cam Ranh Long Beach | Yes |
| (empty) | Adventure, Mountain | Da Lat | Langbiang Mountain | Yes |
| "historical site" | Historical | Hue | Hue Imperial Citadel | Yes |
| "local food street" | Food, Urban | Ho Chi Minh City | Ben Thanh Market area | Yes |
| "jungle trekking" | Adventure, Nature | (none) | Ba Be National Park | Yes |

In all tested cases the top-1 result was judged as relevant. The composite scoring formula worked particularly well when both a query and preferences were provided — label matching pushed highly relevant places up even when the TF-IDF score alone was moderate.

The location boost also had a clear effect: specifying a location effectively buried off-location results due to the ×0.70 multiplier.

**Baseline comparison — TF-IDF only:**
When we disabled the label and rating components and used only TF-IDF cosine similarity, results were noticeably worse for preference-heavy queries (e.g., "Relax + Nature" with no text query). Many results were semantically close in text but wrong in category. The multi-signal approach clearly outperformed pure TF-IDF in these cases.

**Possible metrics:**
- [x] Human evaluation

**Checklist:**
- [x] State the evaluation metric(s).
- [x] Show the main results in a table, chart, or figure.
- [x] Compare at least one baseline or simple method, if possible.
- [x] Explain what the results mean.

---

## Discussion
*Interpret the results and reflect on the project.*

**What worked well:**
The multi-signal ranking formula was the key improvement over a plain TF-IDF baseline. Combining text similarity with label matching and ratings made the recommendations noticeably more useful, especially for vague or preference-only queries. The location boost is also simple but effective — it avoids the frustrating case where the system recommends a great place that is on the other side of the country.

**What did not work well:**
For very short or generic queries like "something fun," TF-IDF was not very discriminative — many places end up with similar cosine similarity scores. Also, with only 315 places, coverage is limited, and users asking about some regions might get few or no relevant results.

**Why the system performed this way:**
The TF-IDF vocabulary (~1,000 terms) is trained on short descriptions, so it cannot handle complex semantic relationships. A word like "fun" may not appear in many descriptions even if the place would be a great fit. The 8-label system partially compensates for this, but it is also coarse.

**Limitations:**
- Small dataset (315 places out of thousands of real destinations in Vietnam)
- No user history means no collaborative filtering is possible
- Labels were manually assigned and might have inconsistencies
- The feedback endpoint exists in the API but user feedback is not yet used to improve rankings
- English-only interface (original data was in Vietnamese)

**What could be improved:**
- Replace TF-IDF with sentence-transformers or a fine-tuned LLM for better semantic similarity
- Expand the dataset to cover more places and regions
- Implement collaborative filtering once user feedback data accumulates
- Add Vietnamese language support

**Checklist:**
- [x] What worked well?
- [x] What did not work well?
- [x] Why do you think the model/system performed this way?
- [x] What are the limitations?
- [x] What could be improved?

---

## Output / Final Output
*Describe what your project produced.*

The project produced the following:

- A complete FastAPI backend (`app/main.py`) with `/recommend`, `/place/{id}`, `/feedback`, and `/health` endpoints
- Source code organized across `app/recommender.py`, `app/ranking.py`, `app/data_loader.py`, `app/models.py`, and `app/nlp_parser.py`
- A static HTML/CSS/JS frontend (`app/static/index.html`) serving as the travel discovery UI
- Pre-trained TF-IDF model serialized as pickle files (`tfidf_model.pkl`, `tfidf_matrix (1).pkl`)
- The cleaned and labeled dataset (`Vietnam_Tourism_Final_8Labels.csv`, `Full_Translated_DataSet_V2.xlsx`)
- This report

**Checklist:**
- [x] Report
- [x] Source code
- [x] Dashboard / visualization, if applicable
- [x] Web / mobile prototype, if applicable
- [x] Chatbot / assistant, if applicable

---

## Conclusion
*Summarize the project in one short section.*

We built S-Shape Explorer, a chatbot-style travel recommendation system for Vietnam tourism. The problem was that finding a relevant destination in Vietnam is not straightforward, especially without local knowledge or a lot of time to research.

We addressed this with a content-based recommendation pipeline that combines TF-IDF similarity, 8-category preference label matching, user ratings, and a location boost — all served via a FastAPI web application with a simple HTML frontend.

The system works well for the scope of the dataset. Top-1 results were relevant in all of our manual test cases, and the multi-signal scoring clearly outperformed pure TF-IDF as a baseline. The main limitation is the dataset size and the relatively simple text model.

Future work could include expanding the dataset, replacing TF-IDF with semantic embeddings, adding collaborative filtering based on accumulated user feedback, and supporting the Vietnamese language.

**Checklist:**
- [x] Restate the problem.
- [x] Summarize the method.
- [x] Summarize the main findings.
- [x] Mention future work.

---

## References
*List all sources used in the project.*

[1] Burke, R. (2002). Hybrid Recommender Systems: Survey and Experiments. *User Modeling and User-Adapted Interaction*, 12(4), 331–370.

[2] Loh, S., Lorenzi, F., Saldaña, R., & Licthnow, D. (2003). A Tourism Recommender System Based on Collaboration and Text Analysis. *Information Technology & Tourism*, 6(3), 157–165.

[3] Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513–523.

[4] scikit-learn Developers. (2024). Feature extraction — Text feature extraction. scikit-learn documentation. https://scikit-learn.org/stable/modules/feature_extraction.html

[5] FastAPI Documentation. (2024). FastAPI — Modern, fast web framework for building APIs. https://fastapi.tiangolo.com/

**Checklist:**
- [x] Papers
- [x] Articles
- [x] Dataset links
- [x] Documentation

---

## Appendix (optional)

**System Architecture Overview:**

```
User Request (query + preferences + location)
        ↓
FastAPI  POST /recommend
        ↓
nlp_parser._extract_location()  →  location from query text
        ↓
PlaceIndex.top_k_similar()  →  TF-IDF cosine similarity  →  candidate pool
        ↓
ranking.rank()  →  weighted score (TF-IDF + label_match + rating) × location_boost
        ↓
Top-k PlaceResult objects with natural-language explanations
        ↓
JSON response  →  Frontend UI
```

**Scoring weights by input case:**

| Case | TF-IDF | Label Match | Rating |
|------|--------|-------------|--------|
| Query + Preferences | 35% | 45% | 20% |
| Preferences only | 0% | 60% | 40% |
| Query only | 65% | 0% | 35% |
| Nothing provided | 50% | 0% | 50% |

**Checklist:**
- [x] Extra tables
- [x] Link to code repository
