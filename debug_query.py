#!/usr/bin/env python3
"""Debug script to test query processing for 'local market Ho Chi Minh city'"""

import sys
sys.path.insert(0, '.')

from app.nlp_parser import _normalise, _extract_location
from app.recommender import PlaceIndex, _tfidf_model
from app.data_loader import PLACES
from app.ranking import rank
import pandas as pd

# Load data
df = pd.read_csv('Vietnam_Tourism_Final_8Labels.csv')
ben_thanh_idx = df[df['Place_Name'].str.contains('Ben Thanh', case=False)].index[0]
print(f"🔍 Ben Thanh Market found at index {ben_thanh_idx}")
print(f"  Name: {df.iloc[ben_thanh_idx]['Place_Name']}")
print(f"  Location: {df.iloc[ben_thanh_idx]['Location']}")
print(f"  Description: {df.iloc[ben_thanh_idx]['Description'][:150]}...")
print(f"  Urban: {df.iloc[ben_thanh_idx]['Urban']}, Food: {df.iloc[ben_thanh_idx]['Food']}")
print()

# Test the query
query = "local market Ho Chi Minh city"
print(f"📝 Original query: '{query}'")

# Step 1: Normalize
normalized = _normalise(query)
print(f"✅ Normalized: '{normalized}'")

# Step 2: Extract location
location = _extract_location(normalized)
print(f"✅ Extracted location: '{location}'")
print()

# Step 3: TF-IDF similarity
print("🔎 Testing TF-IDF similarity...")
q_vec = _tfidf_model.transform([normalized.lower()])
from sklearn.metrics.pairwise import cosine_similarity
from app.recommender import _tfidf_matrix
sims = cosine_similarity(q_vec, _tfidf_matrix)[0]

# Find Ben Thanh's similarity score
ben_thanh_similarity = sims[ben_thanh_idx]
print(f"  Ben Thanh Market TF-IDF similarity: {ben_thanh_similarity:.4f}")

# Get top 10 by TF-IDF
import numpy as np
top_indices = np.argsort(sims)[::-1][:10]
print(f"\n  Top 10 by TF-IDF similarity:")
for rank_pos, idx in enumerate(top_indices, 1):
    place_name = PLACES[idx].name
    sim_score = sims[idx]
    print(f"    {rank_pos}. {place_name:<40} {sim_score:.4f}")

print()
print("=" * 70)
print("Testing recommendation retrieval...")
print("=" * 70)

# Use PlaceIndex to get top_k_similar
index = PlaceIndex(PLACES)
candidates = index.top_k_similar(query, k=10)
print(f"\n✅ PlaceIndex.top_k_similar() returned {len(candidates)} candidates")
for i, (place, score) in enumerate(candidates, 1):
    is_ben_thanh = "⭐ BEN THANH" if place.name == "Ben Thanh Market" else ""
    print(f"  {i}. {place.name:<40} TF-IDF: {score:.4f} {is_ben_thanh}")

print()
print("=" * 70)
print("Testing ranking with Food preference...")
print("=" * 70)

# Rank with Food preference
results = rank(candidates, preferences=["Food"], query_location=location, has_query=True)
print(f"\n✅ Ranking returned {len(results)} results")
for i, result in enumerate(results, 1):
    is_ben_thanh = "⭐ BEN THANH" if result.place.name == "Ben Thanh Market" else ""
    print(f"  {i}. {result.place.name:<40} Score: {result.score:.4f} {is_ben_thanh}")
    if result.place.name == "Ben Thanh Market":
        print(f"     Matched labels: {result.matched_labels}")
        print(f"     Urban: {result.place.urban}, Food: {result.place.food}")
