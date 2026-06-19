"""
visualize_extra.py — 4 additional charts aligned with the DS/AI exam review topics.

Run:    python visualization/visualize_extra.py   (from project root)
        python visualize_extra.py                 (from inside visualization/)
Output: visualization/charts/  — files 7-10

Exam topics covered:
  Chart 7  — ML: Unsupervised learning / Clustering (PCA 2D scatter)
  Chart 8  — Numpy: Aggregation, indexing  / Top keywords per category
  Chart 9  — ML: Supervised scoring / Sample recommendation breakdown
  Chart 10 — Neural Networks: Layers concept / System pipeline diagram
"""

import os
import pickle
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

warnings.simplefilter("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # visualization/
ROOT       = os.path.dirname(SCRIPT_DIR)                  # project root (data files)
OUT        = os.path.join(SCRIPT_DIR, "charts")           # visualization/charts/
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, ROOT)

CSV_PATH  = os.path.join(ROOT, "Vietnam_Tourism_Final_8Labels.csv")
XLSX_PATH = os.path.join(ROOT, "Full_Translated_DataSet_V2.xlsx")
PKL_VEC   = os.path.join(ROOT, "tfidf_model.pkl")
PKL_MAT   = os.path.join(ROOT, "tfidf_matrix (1).pkl")

LABELS = ["Adventure", "Relax", "Rural", "Urban",
          "Mountain", "Historical", "Food", "Nature"]
COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12",
          "#9b59b6", "#1abc9c", "#e67e22", "#27ae60"]

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data ...")
csv_df  = pd.read_csv(CSV_PATH)
xlsx_df = pd.read_excel(XLSX_PATH)

with open(PKL_VEC, "rb") as f:
    tfidf_model = pickle.load(f)
with open(PKL_MAT, "rb") as f:
    tfidf_matrix = pickle.load(f)

ratings = []
for r in xlsx_df["Rating"]:
    m = re.search(r"(\d+\.?\d*)", str(r))
    val = float(m.group(1)) if m else 0.0
    ratings.append(min(val, 5.0) if val <= 5.0 else 0.0)
csv_df["rating_num"] = ratings

vocab_inv = {v: k for k, v in tfidf_model.vocabulary_.items()}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})


# ════════════════════════════════════════════════════════════════════════════════
# 7. PCA 2D Clustering — Unsupervised ML
#    Reduce the 1000-dim TF-IDF vectors to 2D and colour by primary label.
#    Demonstrates: PCA, dimensionality reduction, unsupervised clustering.
# ════════════════════════════════════════════════════════════════════════════════
print("7/10  PCA 2D clustering scatter plot ...")

dense = tfidf_matrix.toarray()
pca   = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(dense)         # shape (315, 2)

# Assign each destination its primary label (first matching label in priority order)
priority = ["Historical", "Mountain", "Nature", "Food",
            "Adventure", "Relax", "Urban", "Rural"]
def primary_label(row):
    for lbl in priority:
        if row[lbl] == 1:
            return lbl
    return "None"

csv_df["primary"] = csv_df.apply(primary_label, axis=1)
label_color = dict(zip(LABELS, COLORS))
label_color["None"] = "#aaaaaa"

fig, ax = plt.subplots(figsize=(11, 7))

for lbl, col in label_color.items():
    mask = csv_df["primary"] == lbl
    if mask.sum() == 0:
        continue
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=col, label=f"{lbl} ({mask.sum()})",
               s=40, alpha=0.75, edgecolors="white", linewidths=0.4)

var_explained = pca.explained_variance_ratio_
ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% variance)", fontsize=11)
ax.set_title("PCA 2D Clustering of 315 Destinations\n(based on TF-IDF description vectors)",
             fontsize=13, fontweight="bold", pad=14)
ax.legend(loc="upper right", fontsize=8, framealpha=0.8, ncol=2)
ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
ax.axvline(0, color="gray", linewidth=0.4, linestyle="--")

total_var = sum(var_explained) * 100
ax.text(0.01, 0.01, f"Total variance explained: {total_var:.1f}%",
        transform=ax.transAxes, fontsize=8, color="gray")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "7_pca_clustering.png"))
plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# 8. Top Keywords per Category — Numpy aggregation (np.mean, np.argsort)
#    For each of the 8 labels: average TF-IDF score across all matching
#    destinations → reveals the most representative terms per category.
# ════════════════════════════════════════════════════════════════════════════════
print("8/10  Top keywords per category ...")

N_TERMS = 8
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for ax, lbl, col in zip(axes, LABELS, COLORS):
    mask  = csv_df[lbl].values == 1
    if mask.sum() == 0:
        ax.set_visible(False)
        continue

    # Numpy aggregation: mean TF-IDF across label's documents
    avg = np.asarray(tfidf_matrix[mask].mean(axis=0)).flatten()
    top_idx   = np.argsort(avg)[::-1][:N_TERMS]          # np.argsort descending
    top_terms = [vocab_inv[i] for i in top_idx]
    top_vals  = avg[top_idx]

    bars = ax.barh(range(N_TERMS), top_vals[::-1], color=col, alpha=0.85,
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(N_TERMS))
    ax.set_yticklabels(top_terms[::-1], fontsize=8)
    ax.set_title(f"{lbl}  (n={mask.sum()})", fontsize=10, fontweight="bold", color=col)
    ax.set_xlabel("Avg TF-IDF", fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.spines["left"].set_visible(False)

fig.suptitle("Top 8 Keywords per Category\n(average TF-IDF score across destinations in each label)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "8_top_keywords_per_category.png"), bbox_inches="tight")
plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# 9. Sample Recommendation Score Breakdown — Supervised scoring
#    Query: "mountain trekking adventure"
#    Show how the 3 components combine for the Top-10 results.
# ════════════════════════════════════════════════════════════════════════════════
print("9/10  Sample recommendation score breakdown ...")

QUERY = "mountain trekking adventure"
PREFS = ["Adventure", "Mountain"]

# --- TF-IDF similarity ---
q_vec = tfidf_model.transform([QUERY.lower()])
sims  = cosine_similarity(q_vec, tfidf_matrix)[0]

# --- Label score ---
def label_score(row, prefs):
    matched = sum(1 for p in prefs if row[p.lower() if p.lower() in row.index else p] == 1)
    return matched / len(prefs) if prefs else 0.0

# --- Build top-10 results ---
top10_idx = np.argsort(sims)[::-1][:10]
results   = []
for i in top10_idx:
    row    = csv_df.iloc[i]
    s_sem  = float(sims[i])
    s_lbl  = sum(1 for p in PREFS if row.get(p, 0) == 1) / len(PREFS)
    s_rat  = min(ratings[i] / 5.0, 1.0) if ratings[i] > 0 else 0.3
    # Case A weights: query + preferences
    combined = 0.35 * s_sem + 0.45 * s_lbl + 0.20 * s_rat
    results.append({
        "name":     row["Place_Name"][:25],
        "sem":      round(s_sem * 0.35, 4),
        "lbl":      round(s_lbl * 0.45, 4),
        "rat":      round(s_rat * 0.20, 4),
        "combined": round(combined, 4),
    })

results.sort(key=lambda x: x["combined"], reverse=True)
names    = [r["name"] for r in results]
s_sem    = [r["sem"]  for r in results]
s_lbl    = [r["lbl"]  for r in results]
s_rat    = [r["rat"]  for r in results]
combined = [r["combined"] for r in results]

y = np.arange(len(results))

fig, ax = plt.subplots(figsize=(11, 6))
b1 = ax.barh(y, s_sem, label="TF-IDF Similarity  (×0.35)", color="#2d6147", alpha=0.9)
b2 = ax.barh(y, s_lbl, left=s_sem, label="Label Match       (×0.45)", color="#c9a044", alpha=0.9)
b3 = ax.barh(y, s_rat, left=[a+b for a,b in zip(s_sem,s_lbl)],
             label="Rating Score      (×0.20)", color="#c8561a", alpha=0.9)

for i, (c, n) in enumerate(zip(combined, names)):
    ax.text(c + 0.003, i, f"{c:.3f}", va="center", fontsize=8, fontweight="bold")

ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.5)
ax.set_xlabel("Combined Score (Case A: query + preferences)")
ax.set_xlim(0, max(combined) * 1.22)
ax.set_title(f'Sample Recommendation Score Breakdown\nQuery: "{QUERY}"  |  Preferences: {PREFS}',
             fontsize=12, fontweight="bold", pad=14)
ax.legend(loc="lower right", fontsize=8.5)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(OUT, "9_recommendation_score_breakdown.png"))
plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# 10. System Pipeline Diagram — Neural Network "Layers" concept
#     Shows each processing stage as a "layer", analogous to a NN forward pass.
# ════════════════════════════════════════════════════════════════════════════════
print("10/10  System pipeline diagram ...")

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14); ax.set_ylim(0, 6)
ax.axis("off")

# --- Define layers (same concept as NN layers) ---
layers = [
    ("INPUT\nLAYER",      "User Query\n+ Preferences\n+ Location",      0.8,  "#34495e"),
    ("LAYER 1\n(Parser)", "NLP Parser\nextract intent\n& location",      2.6,  "#2980b9"),
    ("LAYER 2\n(TF-IDF)", "TF-IDF\nCosine Similarity\n315 docs",         4.4,  "#2d6147"),
    ("LAYER 3\n(Label)",  "Label Match\n8 categories\nbinary score",     6.2,  "#c9a044"),
    ("LAYER 4\n(Rating)", "Rating\nNormalise\n÷ 5.0",                   8.0,  "#e67e22"),
    ("LAYER 5\n(Combine)","Weighted Sum\n35%+45%+20%\n= combined score", 9.8,  "#8e44ad"),
    ("LAYER 6\n(Boost)",  "Location\nBoost ×1.5\n×1.25 / ×0.70",       11.6,  "#c0392b"),
    ("OUTPUT\nLAYER",     "Top-K\nRanked\nResults",                     13.2,  "#1abc9c"),
]

BOX_W, BOX_H = 1.35, 3.8
Y_CENTER = 3.0

for (layer_title, desc, x_center, color) in layers:
    x0 = x_center - BOX_W / 2
    y0 = Y_CENTER - BOX_H / 2

    # Box shadow
    shadow = mpatches.FancyBboxPatch(
        (x0 + 0.05, y0 - 0.05), BOX_W, BOX_H,
        boxstyle="round,pad=0.08", linewidth=0,
        facecolor="gray", alpha=0.18, zorder=1)
    ax.add_patch(shadow)

    # Main box
    box = mpatches.FancyBboxPatch(
        (x0, y0), BOX_W, BOX_H,
        boxstyle="round,pad=0.08", linewidth=1.5,
        edgecolor=color, facecolor=color + "22", zorder=2)
    ax.add_patch(box)

    # Header bar
    header = mpatches.FancyBboxPatch(
        (x0, y0 + BOX_H - 0.9), BOX_W, 0.9,
        boxstyle="round,pad=0.05", linewidth=0,
        facecolor=color, zorder=3)
    ax.add_patch(header)

    # Layer title (in header)
    ax.text(x_center, y0 + BOX_H - 0.44, layer_title,
            ha="center", va="center", fontsize=6.5, fontweight="bold",
            color="white", zorder=4)

    # Description
    ax.text(x_center, y0 + (BOX_H - 0.9) / 2, desc,
            ha="center", va="center", fontsize=7, color="#2c3e50",
            zorder=4, linespacing=1.5)

    # Arrow to next
    if x_center < layers[-1][2]:
        ax.annotate("", xy=(x_center + BOX_W / 2 + 0.28, Y_CENTER),
                    xytext=(x_center + BOX_W / 2 + 0.02, Y_CENTER),
                    arrowprops=dict(arrowstyle="->", color="#555555",
                                   lw=1.5, mutation_scale=14), zorder=5)

# Weight annotations on combine layer
ax.text(9.8, Y_CENTER - BOX_H / 2 - 0.38,
        "35% TF-IDF + 45% Label + 20% Rating",
        ha="center", fontsize=7, color="#8e44ad",
        style="italic")

ax.set_title("S-Shape Explorer — Recommendation System Pipeline\n(Analogous to a Neural Network forward pass)",
             fontsize=13, fontweight="bold", pad=10)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "10_system_pipeline.png"), bbox_inches="tight")
plt.close()


print()
print("Done. 4 extra charts saved to:", OUT)
print("  7_pca_clustering.png                 -- Unsupervised ML: PCA 2D scatter")
print("  8_top_keywords_per_category.png      -- Numpy aggregation: top terms per label")
print("  9_recommendation_score_breakdown.png -- Supervised scoring: component breakdown")
print("  10_system_pipeline.png               -- NN layers concept: system pipeline")
