"""Draws the two schematic figures used in the KNN report (pipeline + KNN voting mechanism)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "reports"
NAVY, BLUE, TEAL, PALE = "#1F3864", "#2F5496", "#1B998B", "#EEF1F8"
plt.rcParams["font.family"] = "serif"

# ---- pipeline diagram
fig, ax = plt.subplots(figsize=(12, 3.6)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 3.6)
ax.text(6, 3.25, "End-to-End Pipeline — K-Nearest Neighbors as the sole data mining technique", ha="center", style="italic", color="#555", fontsize=13)
steps = [("Load Data", "data_loader.py"), ("Preprocess", "clean · encode · scale"), ("Train", "SMOTE + GridSearch"),
         ("Evaluate", "metrics · plots"), ("Predict", "predict.py")]
for i, (t, s) in enumerate(steps):
    x = 0.15 + i * 2.38; hi = i == 2
    ax.add_patch(FancyBboxPatch((x, 0.45), 2.0, 1.95, boxstyle="round,pad=0.02,rounding_size=0.1", fc=NAVY if hi else PALE, ec="#C5CCE0", lw=2))
    ax.add_patch(Circle((x + 1.0, 1.95), 0.25, fc=TEAL if hi else BLUE))
    ax.text(x + 1.0, 1.95, str(i + 1), ha="center", va="center", color="white", fontsize=13, weight="bold")
    ax.text(x + 1.0, 1.4, t, ha="center", va="center", fontsize=14, weight="bold", color="white" if hi else "#1a1a1a")
    ax.text(x + 1.0, 0.8, s, ha="center", va="center", fontsize=9.5, color="#DDE3F3" if hi else "#555")
    if i < 4: ax.annotate("", xy=(x + 2.33, 1.4), xytext=(x + 2.05, 1.4), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2))
fig.savefig(OUT / "pipeline_diagram.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- KNN mechanism diagram
rng = np.random.default_rng(3)
fig, ax = plt.subplots(figsize=(7, 5))
cols = {"Control": "#1B998B", "Benign": "#E0A526", "PDAC": "#C0392B"}
centers = {"Control": (-1.6, -0.8), "Benign": (0.2, 1.4), "PDAC": (1.8, -0.6)}
for k, c in centers.items():
    pts = rng.normal(c, 0.75, (22, 2)); ax.scatter(*pts.T, s=38, c=cols[k], label=k, alpha=.85, edgecolor="white")
q = np.array([0.9, 0.25]); ax.scatter(*q, marker="*", s=380, c=NAVY, label="New patient", zorder=5)
ax.add_patch(Circle(q, 1.0, fill=False, ls="--", ec=NAVY, lw=1.8))
ax.text(q[0], q[1] - 1.25, "k nearest neighbours vote", ha="center", color=NAVY, fontsize=10)
ax.set_title("KNN: classify a sample by the majority class among its k nearest training samples", fontsize=10.5)
ax.set_xlabel("Scaled biomarker 1 (e.g. REG1B)"); ax.set_ylabel("Scaled biomarker 2 (e.g. TFF1)")
ax.legend(loc="upper left", fontsize=9, frameon=False); ax.set_aspect("equal"); fig.tight_layout()
fig.savefig(OUT / "knn_mechanism_diagram.png", dpi=150); plt.close(fig)
