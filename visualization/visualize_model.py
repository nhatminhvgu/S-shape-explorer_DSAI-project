"""
visualize_model.py — Generate charts and matrices for the S-Shape Explorer model.

Run:    python visualization/visualize_model.py   (from project root)
        python visualize_model.py                 (from inside visualization/)
Output: visualization/charts/  — 6 PNG files
"""

import os
import pickle
import re
import warnings

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no GUI required)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

warnings.simplefilter("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # visualization/
ROOT       = os.path.dirname(SCRIPT_DIR)                  # project root (data files)
OUT        = os.path.join(SCRIPT_DIR, "charts")           # visualization/charts/
os.makedirs(OUT, exist_ok=True)

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

# Parse ratings
ratings = []
for r in xlsx_df["Rating"]:
    m = re.search(r"(\d+\.?\d*)", str(r))
    val = float(m.group(1)) if m else 0.0
    ratings.append(min(val, 5.0) if val <= 5.0 else 0.0)
csv_df["rating_num"] = ratings

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})


# ════════════════════════════════════════════════════════════════════════════════
# 1. Label Distribution — Bar chart
# ════════════════════════════════════════════════════════════════════════════════
print("1/6  Label distribution bar chart ...")
counts = [int(csv_df[l].sum()) for l in LABELS]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(LABELS, counts, color=COLORS, edgecolor="white", linewidth=0.8, width=0.6)
ax.set_title("Tourism Label Distribution (315 destinations)", fontsize=14, fontweight="bold", pad=14)
ax.set_ylabel("Number of destinations")
ax.set_ylim(0, max(counts) * 1.18)

for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            str(cnt), ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.axhline(np.mean(counts), color="gray", linestyle="--", linewidth=1,
           label=f"Average: {np.mean(counts):.0f}")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "1_label_distribution.png"))
plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# 2. Label Co-occurrence Matrix — Heatmap
# ════════════════════════════════════════════════════════════════════════════════
print("2/6  Label co-occurrence heatmap ...")
label_matrix = csv_df[LABELS].values.astype(float)
cooc = label_matrix.T @ label_matrix          # shape (8, 8)

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(cooc, cmap="YlOrRd", aspect="auto")
plt.colorbar(im, ax=ax, label="Number of destinations sharing both labels")

ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, rotation=35, ha="right")
ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
ax.set_title("Label Co-occurrence Matrix", fontsize=13, fontweight="bold", pad=14)

for i in range(len(LABELS)):
    for j in range(len(LABELS)):
        val = int(cooc[i, j])
        color = "white" if cooc[i, j] > cooc.max() * 0.6 else "black"
        ax.text(j, i, str(val), ha="center", va="center", fontsize=8.5, color=color, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "2_label_cooccurrence_matrix.png"))
plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# 3. TF-IDF Top-Terms Heatmap (top 20 destinations x top 15 keywords)
# ════════════════════════════════════════════════════════════════════════════════
print("3/6  TF-IDF top-terms heatmap ...")
N_PLACES = 20
N_TERMS  = 15

# Select top N_TERMS by total TF-IDF weight across all documents
term_sums    = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
top_term_idx = np.argsort(term_sums)[::-1][:N_TERMS]
vocab        = {v: k for k, v in tfidf_model.vocabulary_.items()}
top_terms    = [vocab[i] for i in top_term_idx]

# Select top N_PLACES destinations by total TF-IDF weight
place_sums    = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
top_place_idx = np.argsort(place_sums)[::-1][:N_PLACES]
place_names   = [csv_df.iloc[i]["Place_Name"][:28] for i in top_place_idx]

sub_matrix = tfidf_matrix[np.ix_(top_place_idx, top_term_idx)].toarray()

fig, ax = plt.subplots(figsize=(13, 8))
im = ax.imshow(sub_matrix, cmap="Blues", aspect="auto")
plt.colorbar(im, ax=ax, label="TF-IDF score")

ax.set_xticks(range(N_TERMS));  ax.set_xticklabels(top_terms, rotation=40, ha="right", fontsize=9)
ax.set_yticks(range(N_PLACES)); ax.set_yticklabels(place_names, fontsize=8)
ax.set_title(f"TF-IDF Matrix — Top {N_TERMS} Keywords x Top {N_PLACES} Destinations",
             fontsize=13, fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "3_tfidf_heatmap.png"))
plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# 4. Cosine Similarity Matrix (top 15 destinations by TF-IDF weight)
# ════════════════════════════════════════════════════════════════════════════════
print("4/6  Cosine similarity matrix ...")
N_SIM     = 15
top_idx   = np.argsort(place_sums)[::-1][:N_SIM]
sub_mat   = tfidf_matrix[top_idx]
sim_mat   = cosine_similarity(sub_mat)
names_sim = [csv_df.iloc[i]["Place_Name"][:22] for i in top_idx]

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(sim_mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="Cosine similarity")

ax.set_xticks(range(N_SIM)); ax.set_xticklabels(names_sim, rotation=45, ha="right", fontsize=7.5)
ax.set_yticks(range(N_SIM)); ax.set_yticklabels(names_sim, fontsize=7.5)
ax.set_title(f"Cosine Similarity Matrix — Top {N_SIM} Destinations",
             fontsize=13, fontweight="bold", pad=14)

for i in range(N_SIM):
    for j in range(N_SIM):
        color = "white" if sim_mat[i, j] > 0.5 else "black"
        ax.text(j, i, f"{sim_mat[i,j]:.2f}", ha="center", va="center", fontsize=6.5, color=color)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "4_cosine_similarity_matrix.png"))
plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# 5. Rating Distribution — Histogram
# ════════════════════════════════════════════════════════════════════════════════
print("5/6  Rating distribution histogram ...")
valid_ratings = [r for r in ratings if r > 0]

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(valid_ratings, bins=20, color="#2d6147", edgecolor="white", linewidth=0.6, alpha=0.9)

ax.axvline(np.mean(valid_ratings), color="#c8561a", linestyle="--", linewidth=2,
           label=f"Mean: {np.mean(valid_ratings):.2f}")
ax.axvline(np.median(valid_ratings), color="#c9a044", linestyle="-.", linewidth=2,
           label=f"Median: {np.median(valid_ratings):.2f}")

ax.set_title("Rating Distribution — 315 Destinations", fontsize=13, fontweight="bold", pad=14)
ax.set_xlabel("Rating (out of 5.0)")
ax.set_ylabel("Number of destinations")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "5_rating_distribution.png"))
plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# 6. Scoring Weight Breakdown — Stacked bar
# ════════════════════════════════════════════════════════════════════════════════
print("6/6  Scoring weight breakdown ...")
cases = [
    ("Query + Preferences\n(Case A)", 0.35, 0.45, 0.20),
    ("Preferences only\n(Case B)",    0.00, 0.60, 0.40),
    ("Query only\n(Case C)",          0.65, 0.00, 0.35),
    ("No input\n(Case D)",            0.50, 0.00, 0.50),
]
case_labels = [c[0] for c in cases]
w_sem   = [c[1] for c in cases]
w_label = [c[2] for c in cases]
w_rat   = [c[3] for c in cases]

x     = np.arange(len(cases))
width = 0.5

fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x, w_sem,   width, label="TF-IDF Similarity",       color="#2d6147", edgecolor="white")
b2 = ax.bar(x, w_label, width, bottom=w_sem,
            label="Label Match (8 categories)", color="#c9a044", edgecolor="white")
b3 = ax.bar(x, w_rat,   width,
            bottom=[a + b for a, b in zip(w_sem, w_label)],
            label="Rating Score",               color="#c8561a", edgecolor="white")

for bar_group, weights in [(b1, w_sem), (b2, w_label), (b3, w_rat)]:
    for bar, w in zip(bar_group, weights):
        if w > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(w * 100)}%", ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white")

ax.set_xticks(x); ax.set_xticklabels(case_labels, fontsize=9.5)
ax.set_ylabel("Weight")
ax.set_ylim(0, 1.12)
ax.set_title("Recommendation Scoring Formula by Case", fontsize=13, fontweight="bold", pad=14)
ax.legend(loc="upper right", fontsize=9)

ax.text(0.5, -0.17,
        "* Then multiplied by Location Boost: x1.50 (exact match) | x1.25 (partial) | x0.70 (no match)",
        transform=ax.transAxes, ha="center", fontsize=8.5, color="gray", style="italic")

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(os.path.join(OUT, "6_scoring_weights.png"))
plt.close()


print()
print("Done. 6 charts saved to:", OUT)
print("  1_label_distribution.png         -- Tourism label distribution")
print("  2_label_cooccurrence_matrix.png  -- Label co-occurrence matrix")
print("  3_tfidf_heatmap.png              -- TF-IDF matrix (top keywords x top destinations)")
print("  4_cosine_similarity_matrix.png   -- Cosine similarity between destinations")
print("  5_rating_distribution.png        -- Rating distribution histogram")
print("  6_scoring_weights.png            -- Scoring formula weight breakdown")
