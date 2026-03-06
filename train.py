#!/usr/bin/env python3
"""
Word Embeddings training & visualization pipeline.

1. Small toy corpus (10 sentences, 2D embeddings)
   - Co-occurrence matrix SVG
   - Training animation SVG (SMIL, no loop, arrowheads only)
   - Three small-multiple cluster SVGs (chat/verbs/chien)
   - Dot-product similarity SVG
   - Cosine similarity table data

2. Large text corpus (50-dim embeddings + PCA -> 2D)
   - Read plain text file (--text flag, default: odyssee.txt)
   - Training + PCA projection
   - Training animation SVG (subset of words)
   - Word analogy examples (5 max)
   - Top-N most similar words (3 words, 5 closest each)
"""

import os, sys, math, json, csv, re, html, argparse
from collections import defaultdict, Counter
from itertools import combinations
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from tqdm import tqdm

# ── CLI arguments ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Word Embeddings training & visualization pipeline.")
parser.add_argument("--text", default="odyssee.txt",
                    help="Path to a plain text file for Part 2 (large corpus). Default: odyssee.txt")
args = parser.parse_args()

TEXT_PATH = args.text
TEXT_NAME = os.path.splitext(os.path.basename(TEXT_PATH))[0]

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  PART 1 — Small toy corpus (2D)
# ═══════════════════════════════════════════════════════════════

raw_sentences = [
    "Le chat a dormi sur le tapis",
    "Le chien a dormi sur le coussin",
    "Un chat a poursuivi la souris",
    "Un chien a poursuivi le lapin",
    "Le chat est assis sur un tapis",
    "Le chien est assis sur un coussin",
    "La souris a fui devant le chat",
    "La souris a fui devant le chien",
    "Le lapin a fui devant le chien",
    "Le coussin repose sur le tapis",
]

def tokenize(s):
    s = s.replace("\u2019", "'").replace("\u2011", "-")
    tokens = []
    for t in s.split():
        t = t.strip(" .,:;!?()\"")
        t = t.lower()
        if t:
            tokens.append(t)
    return tokens

sentences = [tokenize(s) for s in raw_sentences]

# ── co-occurrence matrix ────────────────────────────────────
stopwords = set([
    "le","la","les","un","une","de","du","des","d'","l'","au","aux",
    "et","est","a","s'est","se","dans","sur","sous","devant","près","avec",
    "en","vers","qui","que","il","elle","ont","sont",
])

vocab = sorted({t for sent in sentences for t in sent if t not in stopwords})
idx = {w: i for i, w in enumerate(vocab)}
N = len(vocab)
cooc = np.zeros((N, N), dtype=int)
window = 2

for sent in sentences:
    toks = [t for t in sent if t in idx]
    for i, w in enumerate(toks):
        for j in range(max(0, i - window), min(len(toks), i + window + 1)):
            if i == j:
                continue
            cooc[idx[w], idx[toks[j]]] += 1

# save CSV
with open(os.path.join(OUTDIR, "cooccurrence_matrix.csv"), "w", newline="", encoding="utf8") as f:
    writer = csv.writer(f)
    writer.writerow([""] + vocab)
    for i, w in enumerate(vocab):
        writer.writerow([w] + cooc[i].tolist())

# ── Word2Vec 2D training ────────────────────────────────────
VECTOR_SIZE = 2
EPOCHS = 2000
SEED = 1

model = Word2Vec(vector_size=VECTOR_SIZE, window=3, min_count=1, sg=1,
                 negative=5, seed=SEED, workers=1)
model.build_vocab(sentences)

words = list(model.wv.index_to_key)
K = len(words)

# Equidistributed initial positions on the unit circle
angles = np.linspace(0, 2 * math.pi, K, endpoint=False)
init_in = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)

model.wv.vectors[:] = init_in.copy()

# Initialize syn1neg
if hasattr(model, "syn1neg"):
    model.syn1neg[:] = (np.random.RandomState(SEED)
                        .randn(*model.syn1neg.shape).astype(np.float32) * 0.01)
elif hasattr(model, "trainables") and hasattr(model.trainables, "syn1neg"):
    model.trainables.syn1neg[:] = (np.random.RandomState(SEED)
                                   .randn(*model.trainables.syn1neg.shape).astype(np.float32) * 0.01)

# Record positions per epoch
positions = []
positions.append(model.wv.vectors.copy())  # epoch 0 (initial)

for ep in range(1, EPOCHS + 1):
    model.train(sentences, total_examples=model.corpus_count, epochs=1)
    pos = model.wv.vectors.copy()
    pos -= pos.mean(axis=0)
    positions.append(pos)

model.save(os.path.join(OUTDIR, "word2vec_2d.model"))

# ── Cosine similarity table ─────────────────────────────────
final_vecs = positions[-1]
word_to_idx = {w: i for i, w in enumerate(words)}

def cos_sim(w1, w2):
    v1 = final_vecs[word_to_idx[w1]]
    v2 = final_vecs[word_to_idx[w2]]
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))

sim_pairs = [
    ("chat", "chien"), ("souris", "lapin"), ("tapis", "coussin"),
    ("poursuivi", "fui"), ("dormi", "assis"), ("chat", "souris"),
]
sim_table = [(w1, w2, cos_sim(w1, w2)) for w1, w2 in sim_pairs]
with open(os.path.join(OUTDIR, "sim_table.json"), "w") as f:
    json.dump(sim_table, f)

# ═══════════════════════════════════════════════════════════════
#  SVG GENERATION HELPERS
# ═══════════════════════════════════════════════════════════════

def svg_head(w, h, extra_attrs=""):
    return f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg"{extra_attrs}>\n'

def svg_tail():
    return "</svg>\n"

def write_svg(path, content):
    with open(path, "w", encoding="utf8") as f:
        f.write(content)

# ═══════════════════════════════════════════════════════════════
#  SVG 1: Dot-product similarity (similar / orthogonal / opposed)
# ═══════════════════════════════════════════════════════════════

dot_svg = svg_head(588, 260)
dot_svg += '<defs>\n'
dot_svg += '<marker id="tri-black" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#111"/></marker>\n'
dot_svg += '<marker id="tri-green" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#286c44"/></marker>\n'
dot_svg += '</defs>\n'

x_centers = [90, 294, 498]
y_center = 120
r = 70
configs = [
    ("similaire", "+0.91", "#286c44", 15, 55, 30, 20),
    ("orthogonal", "0.00", "#ccc", 15, -5, 90, -10),
    ("opposé", "\u22120.87", "#6aad8a", 15, 55, 170, 40),
]

for xc, (lab, val, color, a_angle, b_angle, _, _) in zip(x_centers, configs):
    dot_svg += f'<circle cx="{xc}" cy="{y_center}" r="{r}" fill="none" stroke="#ccc" stroke-width="0.8" stroke-dasharray="3,3"/>\n'

# Similar
a_rad = math.radians(-15)
b_rad = math.radians(-55)
ax, ay = 90 + 65 * math.cos(a_rad), 120 + 65 * math.sin(a_rad)
bx, by = 90 + 65 * math.cos(b_rad), 120 + 65 * math.sin(b_rad)
dot_svg += f'<line x1="90" y1="120" x2="{ax:.1f}" y2="{ay:.1f}" stroke="#111" stroke-width="2" marker-end="url(#tri-black)"/>\n'
dot_svg += f'<text x="{ax+10:.1f}" y="{ay-2:.1f}" fill="#111" font-size="12" font-weight="600" font-family="serif">a</text>\n'
dot_svg += f'<line x1="90" y1="120" x2="{bx:.1f}" y2="{by:.1f}" stroke="#286c44" stroke-width="2" marker-end="url(#tri-green)"/>\n'
dot_svg += f'<text x="{bx+10:.1f}" y="{by-2:.1f}" fill="#286c44" font-size="12" font-weight="600" font-family="serif">b</text>\n'
dot_svg += f'<text x="90" y="{120 + r + 50}" text-anchor="middle" fill="#777" font-size="12" font-family="serif">similaire</text>\n'
dot_svg += f'<text x="90" y="{120 - r - 18}" text-anchor="middle" fill="#286c44" font-size="14" font-weight="700" font-family="serif">+0.91</text>\n'

# Orthogonal
a_rad2 = math.radians(-10)
b_rad2 = math.radians(-100)
ax2, ay2 = 294 + 65 * math.cos(a_rad2), 120 + 65 * math.sin(a_rad2)
bx2, by2 = 294 + 65 * math.cos(b_rad2), 120 + 65 * math.sin(b_rad2)
dot_svg += f'<line x1="294" y1="120" x2="{ax2:.1f}" y2="{ay2:.1f}" stroke="#111" stroke-width="2" marker-end="url(#tri-black)"/>\n'
dot_svg += f'<text x="{ax2+10:.1f}" y="{ay2-2:.1f}" fill="#111" font-size="12" font-weight="600" font-family="serif">a</text>\n'
dot_svg += f'<line x1="294" y1="120" x2="{bx2:.1f}" y2="{by2:.1f}" stroke="#286c44" stroke-width="2" marker-end="url(#tri-green)"/>\n'
dot_svg += f'<text x="{bx2+10:.1f}" y="{by2-2:.1f}" fill="#286c44" font-size="12" font-weight="600" font-family="serif">b</text>\n'
dot_svg += f'<text x="294" y="{120 + r + 50}" text-anchor="middle" fill="#777" font-size="12" font-family="serif">orthogonal</text>\n'
dot_svg += f'<text x="294" y="{120 - r - 18}" text-anchor="middle" fill="#ccc" font-size="14" font-weight="700" font-family="serif">0.00</text>\n'

# Opposed
a_rad3 = math.radians(-30)
b_rad3 = math.radians(155)
ax3, ay3 = 498 + 65 * math.cos(a_rad3), 120 + 65 * math.sin(a_rad3)
bx3, by3 = 498 + 65 * math.cos(b_rad3), 120 + 65 * math.sin(b_rad3)
dot_svg += f'<line x1="498" y1="120" x2="{ax3:.1f}" y2="{ay3:.1f}" stroke="#111" stroke-width="2" marker-end="url(#tri-black)"/>\n'
dot_svg += f'<text x="{ax3+10:.1f}" y="{ay3-2:.1f}" fill="#111" font-size="12" font-weight="600" font-family="serif">a</text>\n'
dot_svg += f'<line x1="498" y1="120" x2="{bx3:.1f}" y2="{by3:.1f}" stroke="#286c44" stroke-width="2" marker-end="url(#tri-green)"/>\n'
dot_svg += f'<text x="{bx3-12:.1f}" y="{by3-2:.1f}" fill="#286c44" font-size="12" font-weight="600" font-family="serif" text-anchor="end">b</text>\n'
dot_svg += f'<text x="498" y="{120 + r + 50}" text-anchor="middle" fill="#777" font-size="12" font-family="serif">opposé</text>\n'
dot_svg += f'<text x="498" y="{120 - r - 18}" text-anchor="middle" fill="#6aad8a" font-size="14" font-weight="700" font-family="serif">\u22120.87</text>\n'

dot_svg += svg_tail()
write_svg(os.path.join(OUTDIR, "dot_similarity.svg"), dot_svg)

# ═══════════════════════════════════════════════════════════════
#  SVG 2: Co-occurrence matrix heatmap
# ═══════════════════════════════════════════════════════════════

# Color categories for the matrix
# souris/lapin=blue, chat/chien=grey, tapis/coussin=bordeaux, rest=grey/black
word_categories = {
    "chat": "#286c44", "chien": "#286c44",
}

maxc = cooc.max() if cooc.size else 1
cell = 36
pad_left = 100
pad_top = 80
W2 = pad_left + N * cell + 20
H2 = pad_top + N * cell + 20
cooc_svg = svg_head(W2, H2)

def color_for_count(c):
    if c == 0:
        return "rgba(0,0,0,0.02)"
    t = c / maxc
    opacity = 0.3 + 0.5 * t
    return f"rgba(40,108,68,{opacity:.2f})"

for i in range(N):
    for j in range(N):
        x = pad_left + j * cell
        y = pad_top + i * cell
        c = cooc[i, j]
        col = color_for_count(c)
        cooc_svg += f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="3" fill="{col}"/>\n'
        if c > 0:
            cooc_svg += f'<text x="{x + cell/2}" y="{y + cell/2 + 4}" font-family="serif" font-size="11" fill="#fcfcfc" text-anchor="middle">{c}</text>\n'

# Row labels
for i, w in enumerate(vocab):
    y = pad_top + i * cell + cell / 2 + 4
    col = word_categories.get(w, "#111")
    cooc_svg += f'<text x="{pad_left - 8}" y="{y}" font-family="serif" font-size="11" fill="{col}" text-anchor="end">{w}</text>\n'

# Column labels (rotated)
for j, w in enumerate(vocab):
    x = pad_left + j * cell + cell / 2
    col = word_categories.get(w, "#111")
    cooc_svg += f'<text x="{x}" y="{pad_top - 8}" font-family="serif" font-size="11" fill="{col}" text-anchor="start" transform="rotate(-50 {x} {pad_top - 8})">{w}</text>\n'

cooc_svg += svg_tail()
write_svg(os.path.join(OUTDIR, "cooccurrence_matrix.svg"), cooc_svg)

# Keep final positions for later data export
final_pos = positions[-1]

# ═══════════════════════════════════════════════════════════════
#  SVG 3: Three small multiples — vectors on unit circle
#  Style matches dot_similarity.svg: dashed circle, arrows from
#  center, arrowhead markers, labels near tips.
#  Hardcoded angles to clearly show chat≈chien via shared verbs.
# ═══════════════════════════════════════════════════════════════

# Angles (degrees, counter-clockwise from right) chosen so that:
#  - Left to right order: chien, verbs, chat
#  - chat and chien bracket the verbs (close to them, close to each other)
#  - Tight angles between all four (~15° between neighbors)
cluster_angles = {
    "chien":      -20,
    "dormi":      -35,
    "poursuivi":  -50,
    "chat":       -65,
}

animals = {"chat", "chien", "souris", "lapin"}
verbs_set = {"dormi", "assis", "poursuivi", "fui", "repose"}
surfaces = {"tapis", "coussin"}

def build_cluster_svg(angles, color_fn, subtitle=""):
    W, H = 260, 290
    cx, cy, rad = W / 2, H / 2 - 15, 85
    arrow_len = rad * 0.9
    s = svg_head(W, H)
    # Defs: one marker per color used
    colors_used = set(color_fn(w) for w in angles)
    s += '<defs>\n'
    for col in colors_used:
        cid = col.replace("#", "c")
        s += f'<marker id="tri-{cid}" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="{col}"/></marker>\n'
    s += '</defs>\n'
    # Dashed unit circle
    s += f'<circle cx="{cx}" cy="{cy}" r="{rad}" fill="none" stroke="#ccc" stroke-width="0.8" stroke-dasharray="3,3"/>\n'

    for w, angle_deg in angles.items():
        col = color_fn(w)
        cid = col.replace("#", "c")
        a_rad = math.radians(angle_deg)
        tip_x = cx + arrow_len * math.cos(a_rad)
        tip_y = cy + arrow_len * math.sin(a_rad)
        s += f'<line x1="{cx}" y1="{cy}" x2="{tip_x:.1f}" y2="{tip_y:.1f}" stroke="{col}" stroke-width="2" marker-end="url(#tri-{cid})"/>\n'
        # Label: push outward beyond the arrow tip
        lab_r = rad + 14
        lab_x = cx + lab_r * math.cos(a_rad)
        lab_y = cy + lab_r * math.sin(a_rad) + 4
        anchor = "start" if math.cos(a_rad) > 0.2 else ("end" if math.cos(a_rad) < -0.2 else "middle")
        s += f'<text x="{lab_x:.1f}" y="{lab_y:.1f}" text-anchor="{anchor}" fill="{col}" font-size="12" font-weight="600" font-family="serif">{w}</text>\n'

    if subtitle:
        s += f'<text x="{cx}" y="{H - 8}" text-anchor="middle" fill="#777" font-size="9.5" font-family="serif">{subtitle}</text>\n'
    s += svg_tail()
    return s

# Variant 1: chat grey, verbs black, chien light gray
def cv1(w):
    if w == "chat": return "#286c44"
    if w == "chien": return "#ccc"
    return "#111"

# Variant 2: chat light gray, verbs black, chien grey
def cv2(w):
    if w == "chien": return "#286c44"
    if w == "chat": return "#ccc"
    return "#111"

# Variant 3: chat & chien grey, verbs greyed
def cv3(w):
    if w in ("chat", "chien"): return "#286c44"
    return "#ccc"

write_svg(os.path.join(OUTDIR, "cluster_chat.svg"),
          build_cluster_svg(cluster_angles, cv1, "chat co-occurre avec dormi et poursuivi"))
write_svg(os.path.join(OUTDIR, "cluster_chien.svg"),
          build_cluster_svg(cluster_angles, cv2, "chien aussi — mêmes verbes"))
write_svg(os.path.join(OUTDIR, "cluster_both.svg"),
          build_cluster_svg(cluster_angles, cv3, "donc chat ≈ chien, sans jamais co-occurrir"))

# ── SVG : vecteur exemple [1.2, -3.5] ──────────────────────────────────────
vec_ex = [1.2, -3.5]
vec_ex_mag = math.sqrt(vec_ex[0] ** 2 + vec_ex[1] ** 2)
vec_ex_norm = [vec_ex[0] / vec_ex_mag, vec_ex[1] / vec_ex_mag]

W_ve, H_ve = 200, 230
cx_ve, cy_ve, r_ve = W_ve / 2, H_ve / 2 - 5, 70
arrow_len_ve = r_ve * 0.9

ve_svg = svg_head(W_ve, H_ve)
ve_svg += '<defs><marker id="vec-arr" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto">'
ve_svg += '<path d="M0,0 L8,4 L0,8 Z" fill="#286c44"/></marker></defs>\n'
ve_svg += f'<circle cx="{cx_ve}" cy="{cy_ve}" r="{r_ve}" fill="none" stroke="#ccc" stroke-width="0.8" stroke-dasharray="3,3"/>\n'

# SVG : y-axis est inversé par rapport aux coordonnées mathématiques
tip_x = cx_ve + arrow_len_ve * vec_ex_norm[0]
tip_y = cy_ve - arrow_len_ve * vec_ex_norm[1]
ve_svg += f'<line x1="{cx_ve}" y1="{cy_ve}" x2="{tip_x:.1f}" y2="{tip_y:.1f}" stroke="#286c44" stroke-width="2" marker-end="url(#vec-arr)"/>\n'

# Label "chat" près de la pointe
lab_off = 8 if vec_ex_norm[0] >= 0 else -8
lab_anchor = "start" if vec_ex_norm[0] >= 0 else "end"
ve_svg += f'<text x="{tip_x + lab_off:.1f}" y="{tip_y + 4:.1f}" fill="#286c44" font-size="12" font-weight="600" font-family="serif" text-anchor="{lab_anchor}">chat</text>\n'

# Légende
ve_svg += f'<text x="{cx_ve}" y="{H_ve - 10}" text-anchor="middle" fill="#777" font-size="10" font-family="serif">[0,2\u2009\u22123,5]</text>\n'
ve_svg += svg_tail()
write_svg(os.path.join(OUTDIR, "vector_example.svg"), ve_svg)

# ═══════════════════════════════════════════════════════════════
#  SVG 4: Training animation (SMIL, no loop, arrowheads only)
# ═══════════════════════════════════════════════════════════════

# We'll sample epochs to keep the SVG manageable
# Take ~100 frames spread across all epochs
n_frames = 100
frame_indices = np.linspace(0, len(positions) - 1, n_frames, dtype=int)
sampled_positions = [positions[i] for i in frame_indices]

W_train = 460
H_train = 380
center_train = (W_train / 2, H_train / 2 - 10)
allpos = np.stack(sampled_positions)
scale_train = 140.0 / max(1e-9, np.abs(allpos).max())
dur_train = "15s"

# Color mapping for the toy animation
WORD_COLORS = {
    "souris": "#111", "lapin": "#111",
    "chat": "#286c44", "chien": "#286c44",
    "tapis": "#111", "coussin": "#111",
}
highlight_words = {"chat", "chien", "souris", "lapin", "tapis", "coussin"}

train_svg = svg_head(W_train, H_train)
train_svg += f'<rect x="0" y="0" width="{W_train}" height="{H_train}" fill="#fcfcfc" rx="4"/>\n'

# Reference circles
for ri in range(1, 5):
    rad = ri * scale_train * 0.5
    if rad < 5 or rad > 200:
        continue
    train_svg += f'<circle cx="{center_train[0]}" cy="{center_train[1]}" r="{rad:.1f}" fill="none" stroke="#ccc" stroke-width="0.5"/>\n'

# Axes
train_svg += f'<line x1="15" y1="{center_train[1]}" x2="{W_train-15}" y2="{center_train[1]}" stroke="#ccc" stroke-width="0.4"/>\n'
train_svg += f'<line x1="{center_train[0]}" y1="15" x2="{center_train[0]}" y2="{H_train-30}" stroke="#ccc" stroke-width="0.4"/>\n'

# Arrow markers
# Arrow markers for each color used
used_colors = set(WORD_COLORS.values()) | {"#ccc"}
train_svg += '<defs>\n'
for c in used_colors:
    cid = c.replace("#", "c")
    train_svg += f'<marker id="arr-{cid}" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="{c}"/></marker>\n'
train_svg += '</defs>\n'

# Draw non-highlighted words first (behind), then highlighted on top
for is_highlight_pass in [False, True]:
    for i, w in enumerate(words):
        is_hl = w in highlight_words
        if is_hl != is_highlight_pass:
            continue

        vals_x = []
        vals_y = []
        for frame in sampled_positions:
            vx, vy = frame[i]
            sx = center_train[0] + vx * scale_train
            sy = center_train[1] - vy * scale_train  # flip y
            vals_x.append(f"{sx:.1f}")
            vals_y.append(f"{sy:.1f}")

        values_x = ";".join(vals_x)
        values_y = ";".join(vals_y)

        if is_hl:
            col = WORD_COLORS.get(w, "#777")
            opacity, sw = "1", "1.8"
        else:
            col = "#ccc"
            opacity, sw = "0.5", "1"
        cid = col.replace("#", "c")
        marker_id = f"arr-{cid}"

        # Line from center to tip (animated), arrowhead only
        train_svg += f'<line x1="{center_train[0]}" y1="{center_train[1]}" x2="{vals_x[0]}" y2="{vals_y[0]}" stroke="{col}" stroke-width="{sw}" opacity="{opacity}" marker-end="url(#{marker_id})">\n'
        train_svg += f'  <animate attributeName="x2" values="{values_x}" dur="{dur_train}" fill="freeze"/>\n'
        train_svg += f'  <animate attributeName="y2" values="{values_y}" dur="{dur_train}" fill="freeze"/>\n'
        train_svg += '</line>\n'

        # Label at tip (animated)
        dx = float(vals_x[-1]) - center_train[0]
        if w == "chien":
            anchor, offset_x, offset_y = "start", 8, 2
        elif w == "chat":
            anchor, offset_x, offset_y = "end", -8, -2
        else:
            anchor = "start" if dx >= 0 else "end"
            offset_x = 6 if dx >= 0 else -6
            offset_y = -3

        label_x = ";".join([f"{float(v) + offset_x:.1f}" for v in vals_x])
        label_y = ";".join([f"{float(v) + offset_y:.1f}" for v in vals_y])

        font_size = "10" if is_hl else "8"
        train_svg += f'<text x="{float(vals_x[0]) + offset_x:.1f}" y="{float(vals_y[0]) - 3:.1f}" fill="{col}" font-size="{font_size}" font-weight="600" font-family="serif" text-anchor="{anchor}" opacity="{opacity}">\n'
        train_svg += f'  <animate attributeName="x" values="{label_x}" dur="{dur_train}" fill="freeze"/>\n'
        train_svg += f'  <animate attributeName="y" values="{label_y}" dur="{dur_train}" fill="freeze"/>\n'
        train_svg += f'{w}</text>\n'

# Epoch counter
n_epoch_labels = 20
epoch_label_indices = np.linspace(0, EPOCHS, n_epoch_labels + 1, dtype=int)
label_dur = float(dur_train.replace("s", "")) / (n_epoch_labels + 1)
for li, ep_num in enumerate(epoch_label_indices):
    begin = f"{li * label_dur:.2f}s"
    train_svg += f'<text x="{W_train - 15}" y="25" text-anchor="end" fill="#ccc" font-size="11" font-family="serif" opacity="0">époque {ep_num}\n'
    train_svg += f'  <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.05;0.95;1" begin="{begin}" dur="{label_dur:.2f}s" fill="freeze"/>\n'
    train_svg += '</text>\n'

train_svg += svg_tail()
write_svg(os.path.join(OUTDIR, "embedding_train_animation.svg"), train_svg)

# ═══════════════════════════════════════════════════════════════
#  Export small corpus data as JSON for HTML
# ═══════════════════════════════════════════════════════════════

small_data = {
    "words": words,
    "vocab": vocab,
    "cooc": cooc.tolist(),
    "sim_table": sim_table,
    "final_positions": {w: final_pos[i].tolist() for i, w in enumerate(words)},
}
with open(os.path.join(OUTDIR, "small_corpus_data.json"), "w") as f:
    json.dump(small_data, f, ensure_ascii=False)

print("[OK] Small corpus SVGs and data generated.")

# ═══════════════════════════════════════════════════════════════
#  PART 2 — Large text corpus (50-dim + PCA)
# ═══════════════════════════════════════════════════════════════

print(f"[...] Reading text from {TEXT_PATH}...")
with open(TEXT_PATH, "r", encoding="utf-8") as f:
    corpus_text = f.read()
print(f"[OK] Read {len(corpus_text)} characters from {TEXT_PATH}.")

# Save cleaned text (cap at 50k chars for display)
with open(os.path.join(OUTDIR, "corpus_text.txt"), "w", encoding="utf-8") as f:
    f.write(corpus_text[:50000])

# Tokenize using NLTK French tokenizer
import nltk
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize

print("[...] Tokenizing with NLTK (French)...")

corpus_sentences_raw = sent_tokenize(corpus_text, language='french')
corpus_sentences = []
for sent in corpus_sentences_raw:
    tokens = word_tokenize(sent.lower(), language='french')
    # Keep only alphabetic tokens of length > 1
    tokens = [t for t in tokens if t.isalpha() and len(t) > 1]
    if len(tokens) >= 3:
        corpus_sentences.append(tokens)

print(f"[OK] {TEXT_NAME}: {len(corpus_sentences)} sentences, {sum(len(s) for s in corpus_sentences)} tokens")

# ── Train Word2Vec 50-dim on text corpus ─────────────────────

CORPUS_VECTOR_SIZE = 50
CORPUS_EPOCHS = 100
CORPUS_SEED = 42

corpus_model = Word2Vec(
    vector_size=CORPUS_VECTOR_SIZE, window=5, min_count=3, sg=1,
    negative=10, seed=CORPUS_SEED, workers=1
)
corpus_model.build_vocab(corpus_sentences)

corpus_words = list(corpus_model.wv.index_to_key)
corpus_K = len(corpus_words)
print(f"[OK] {TEXT_NAME} vocabulary: {corpus_K} words")

# Equidistributed init on unit circle (in first 2 dims, rest small random)
rng = np.random.RandomState(CORPUS_SEED)
corpus_angles = np.linspace(0, 2 * math.pi, corpus_K, endpoint=False)
corpus_init = rng.randn(corpus_K, CORPUS_VECTOR_SIZE).astype(np.float32) * 0.01
corpus_init[:, 0] = np.cos(corpus_angles)
corpus_init[:, 1] = np.sin(corpus_angles)
corpus_model.wv.vectors[:] = corpus_init.copy()

if hasattr(corpus_model, "syn1neg"):
    corpus_model.syn1neg[:] = rng.randn(*corpus_model.syn1neg.shape).astype(np.float32) * 0.01
elif hasattr(corpus_model, "trainables") and hasattr(corpus_model.trainables, "syn1neg"):
    corpus_model.trainables.syn1neg[:] = rng.randn(*corpus_model.trainables.syn1neg.shape).astype(np.float32) * 0.01

# Record positions per epoch (PCA-projected)
corpus_positions = []
# Save initial
corpus_positions.append(corpus_model.wv.vectors.copy())

for ep in range(1, CORPUS_EPOCHS + 1):
    corpus_model.train(corpus_sentences, total_examples=corpus_model.corpus_count, epochs=1)
    corpus_positions.append(corpus_model.wv.vectors.copy())

corpus_model.save(os.path.join(OUTDIR, "word2vec_corpus.model"))
print(f"[OK] {TEXT_NAME} model trained.")

# ── PCA projection ──────────────────────────────────────────

# Fit PCA on final vectors
pca = PCA(n_components=2, random_state=CORPUS_SEED)
corpus_final_2d = pca.fit_transform(corpus_positions[-1])

# Project all epochs with same PCA
corpus_positions_2d = [pca.transform(pos) for pos in corpus_positions]

# Explained variance
explained_var = pca.explained_variance_ratio_
print(f"[OK] PCA explained variance: {explained_var[0]:.3f}, {explained_var[1]:.3f} (total: {sum(explained_var):.3f})")

# Save PCA info
pca_info = {
    "explained_variance": explained_var.tolist(),
    "n_components": 2,
    "n_original_dims": CORPUS_VECTOR_SIZE,
}
with open(os.path.join(OUTDIR, "pca_info.json"), "w") as f:
    json.dump(pca_info, f)

# ── Select words for corpus animation ─────────────────────────

# Use NLTK's French stopword list
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords as nltk_stopwords
stopwords_fr = set(nltk_stopwords.words('french'))

K_FILTER = 300
all_candidates = [w for w in corpus_words[:K_FILTER]
                  if w in corpus_model.wv and len(w) > 3 and w not in stopwords_fr]

# Use curated highlight words relevant to the Odyssey
PREFERRED_HIGHLIGHTS = ["odysseus", "mer", "divin", "écumeuse", "dieux",
                         "immortels"]
main_highlight = [w for w in PREFERRED_HIGHLIGHTS if w in corpus_model.wv][:5]

# Fallback if not enough words found
if len(main_highlight) < 5:
    extra = [w for w in all_candidates if w not in main_highlight][:5 - len(main_highlight)]
    main_highlight.extend(extra)

print(f"[OK] Selected highlight words: {main_highlight}")

# Pick ~50 background words from top frequent content words
bg_words = [w for w in all_candidates if w not in main_highlight][:50]
# If not enough, add from top frequency words
if len(bg_words) < 30:
    freq_words = [w for w in corpus_words[:80] if w not in main_highlight and w not in bg_words]
    bg_words.extend(freq_words[:50 - len(bg_words)])

all_anim_words = main_highlight + bg_words
all_anim_indices = [corpus_words.index(w) for w in all_anim_words if w in corpus_words]
all_anim_words = [corpus_words[i] for i in all_anim_indices]

print(f"[OK] {TEXT_NAME} animation: {len(main_highlight)} highlighted, {len(bg_words)} background")

# ── Corpus training animation SVG ─────────────────────────────

n_corpus_frames = 60
corpus_frame_indices = np.linspace(0, len(corpus_positions_2d) - 1, n_corpus_frames, dtype=int)
corpus_sampled = [corpus_positions_2d[i] for i in corpus_frame_indices]

W_ep = 600
H_ep = 500
center_ep = (W_ep / 2, H_ep / 2)
ep_allpos = np.stack([p[all_anim_indices] for p in corpus_sampled])
ep_scale = 180.0 / max(1e-9, np.abs(ep_allpos).max())
dur_ep = "20s"

ep_train_svg = svg_head(W_ep, H_ep)
ep_train_svg += f'<rect x="0" y="0" width="{W_ep}" height="{H_ep}" fill="#fcfcfc" rx="4"/>\n'
ep_train_svg += f'<line x1="15" y1="{center_ep[1]}" x2="{W_ep-15}" y2="{center_ep[1]}" stroke="#ccc" stroke-width="0.4"/>\n'
ep_train_svg += f'<line x1="{center_ep[0]}" y1="15" x2="{center_ep[0]}" y2="{H_ep-15}" stroke="#ccc" stroke-width="0.4"/>\n'

ep_train_svg += '<defs>\n'
ep_train_svg += '<marker id="narr-g" markerWidth="5" markerHeight="5" refX="5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="#286c44" opacity="0.7"/></marker>\n'
ep_train_svg += '<marker id="narr-b" markerWidth="4" markerHeight="4" refX="4" refY="2" orient="auto"><path d="M0,0 L4,2 L0,4 Z" fill="#ccc" opacity="0.5"/></marker>\n'
ep_train_svg += '</defs>\n'

for idx_in_anim, word_idx in enumerate(all_anim_indices):
    w = corpus_words[word_idx]
    is_main = w in main_highlight

    vals_x = []
    vals_y = []
    for frame in corpus_sampled:
        vx, vy = frame[word_idx]
        sx = center_ep[0] + vx * ep_scale
        sy = center_ep[1] - vy * ep_scale
        vals_x.append(f"{sx:.1f}")
        vals_y.append(f"{sy:.1f}")

    values_x = ";".join(vals_x)
    values_y = ";".join(vals_y)

    if is_main:
        col = "#286c44"
        marker = "narr-g"
        sw = "1.3"
        fs = "10"
        fw = "600"
        opacity = "0.8"
    else:
        col = "#ccc"
        marker = "narr-b"
        sw = "0.6"
        fs = "0"  # no label for background
        fw = "400"
        opacity = "0.3"

    # Line
    ep_train_svg += f'<line x1="{center_ep[0]}" y1="{center_ep[1]}" x2="{vals_x[0]}" y2="{vals_y[0]}" stroke="{col}" stroke-width="{sw}" opacity="{opacity}" marker-end="url(#{marker})">\n'
    ep_train_svg += f'  <animate attributeName="x2" values="{values_x}" dur="{dur_ep}" fill="freeze"/>\n'
    ep_train_svg += f'  <animate attributeName="y2" values="{values_y}" dur="{dur_ep}" fill="freeze"/>\n'
    ep_train_svg += '</line>\n'

    # Small dot for background words
    if not is_main:
        ep_train_svg += f'<circle cx="{vals_x[0]}" cy="{vals_y[0]}" r="1.5" fill="{col}" opacity="0.3">\n'
        ep_train_svg += f'  <animate attributeName="cx" values="{values_x}" dur="{dur_ep}" fill="freeze"/>\n'
        ep_train_svg += f'  <animate attributeName="cy" values="{values_y}" dur="{dur_ep}" fill="freeze"/>\n'
        ep_train_svg += '</circle>\n'

    # Label for main words only
    if is_main:
        dx = float(vals_x[-1]) - center_ep[0]
        anchor = "start" if dx >= 0 else "end"
        off_x = 7 if dx >= 0 else -7
        label_x = ";".join([f"{float(v) + off_x:.1f}" for v in vals_x])
        label_y = ";".join([f"{float(v) - 4:.1f}" for v in vals_y])
        ep_train_svg += f'<text x="{float(vals_x[0]) + off_x:.1f}" y="{float(vals_y[0]) - 4:.1f}" fill="{col}" font-size="{fs}" font-weight="{fw}" font-family="serif" text-anchor="{anchor}">\n'
        ep_train_svg += f'  <animate attributeName="x" values="{label_x}" dur="{dur_ep}" fill="freeze"/>\n'
        ep_train_svg += f'  <animate attributeName="y" values="{label_y}" dur="{dur_ep}" fill="freeze"/>\n'
        ep_train_svg += f'{w}</text>\n'

# Epoch counter
ep_n_labels = 10
ep_epoch_labels = np.linspace(0, CORPUS_EPOCHS, ep_n_labels + 1, dtype=int)
ep_label_dur = float(dur_ep.replace("s", "")) / (ep_n_labels + 1)
for li, ep_num in enumerate(ep_epoch_labels):
    begin = f"{li * ep_label_dur:.2f}s"
    ep_train_svg += f'<text x="{W_ep - 15}" y="25" text-anchor="end" fill="#ccc" font-size="11" font-family="serif" opacity="0">époque {ep_num}\n'
    ep_train_svg += f'  <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.05;0.95;1" begin="{begin}" dur="{ep_label_dur:.2f}s" fill="freeze"/>\n'
    ep_train_svg += '</text>\n'

ep_train_svg += svg_tail()
write_svg(os.path.join(OUTDIR, "corpus_train_animation.svg"), ep_train_svg)

# ── Final PCA scatter SVG ────────────────────────────────────

W_pca = 600
H_pca = 500
center_pca = (W_pca / 2, H_pca / 2)
pca_scale = 180.0 / max(1e-9, np.abs(corpus_final_2d[all_anim_indices]).max())

pca_svg = svg_head(W_pca, H_pca)
pca_svg += f'<rect x="0" y="0" width="{W_pca}" height="{H_pca}" fill="#fcfcfc" rx="4"/>\n'
pca_svg += f'<line x1="15" y1="{center_pca[1]}" x2="{W_pca-15}" y2="{center_pca[1]}" stroke="#ccc" stroke-width="0.4"/>\n'
pca_svg += f'<line x1="{center_pca[0]}" y1="15" x2="{center_pca[0]}" y2="{H_pca-15}" stroke="#ccc" stroke-width="0.4"/>\n'

# PCA axis labels
pca_svg += f'<text x="{W_pca - 20}" y="{center_pca[1] + 16}" fill="#ccc" font-size="10" font-family="serif" text-anchor="end">PC1 ({explained_var[0]*100:.0f}%)</text>\n'
pca_svg += f'<text x="{center_pca[0] + 8}" y="22" fill="#ccc" font-size="10" font-family="serif">PC2 ({explained_var[1]*100:.0f}%)</text>\n'

# Collect dots and labels, then nudge labels to avoid overlaps
pca_dots = []
pca_labels = []
for idx_in_anim, word_idx in enumerate(all_anim_indices):
    w = corpus_words[word_idx]
    is_main = w in main_highlight
    vx, vy = corpus_final_2d[word_idx]
    sx = center_pca[0] + vx * pca_scale
    sy = center_pca[1] - vy * pca_scale

    if is_main:
        pca_dots.append((sx, sy, "3.5", "#286c44", "0.8"))
        dx = sx - center_pca[0]
        anchor = "start" if dx >= 0 else "end"
        off = 7 if dx >= 0 else -7
        pca_labels.append({"w": w, "lx": sx + off, "ly": sy + 3, "lw": len(w) * 6.5, "anchor": anchor})
    else:
        pca_dots.append((sx, sy, "1.8", "#ccc", "0.4"))

# Nudge main-highlight labels apart
def _bbox(d):
    if d["anchor"] == "start":
        x0, x1 = d["lx"], d["lx"] + d["lw"]
    else:
        x0, x1 = d["lx"] - d["lw"], d["lx"]
    return x0, d["ly"] - 10, x1, d["ly"] + 3

def _overlap(a, b):
    ax0, ay0, ax1, ay1 = _bbox(a)
    bx0, by0, bx1, by1 = _bbox(b)
    return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0

for _ in range(30):
    moved = False
    for i in range(len(pca_labels)):
        for j in range(i + 1, len(pca_labels)):
            if _overlap(pca_labels[i], pca_labels[j]):
                if pca_labels[i]["ly"] <= pca_labels[j]["ly"]:
                    pca_labels[i]["ly"] -= 4
                    pca_labels[j]["ly"] += 4
                else:
                    pca_labels[i]["ly"] += 4
                    pca_labels[j]["ly"] -= 4
                moved = True
    if not moved:
        break

# Render dots
for sx, sy, r, col, op in pca_dots:
    pca_svg += f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{r}" fill="{col}" opacity="{op}"/>\n'
# Render labels
for d in pca_labels:
    pca_svg += f'<text x="{d["lx"]:.1f}" y="{d["ly"]:.1f}" fill="#286c44" font-size="11" font-weight="600" font-family="serif" text-anchor="{d["anchor"]}">{d["w"]}</text>\n'

pca_svg += svg_tail()
write_svg(os.path.join(OUTDIR, "corpus_pca_scatter.svg"), pca_svg)

# ═══════════════════════════════════════════════════════════════
#  Word analogies & top-N similar
# ═══════════════════════════════════════════════════════════════

def word_analogy(model, a, b, c, topn=5):
    """a is to b as c is to ?"""
    try:
        results = model.wv.most_similar(positive=[b, c], negative=[a], topn=topn)
        return results
    except KeyError:
        return []

def top_similar(model, word, topn=8):
    try:
        return model.wv.most_similar(word, topn=topn)
    except KeyError:
        return []

# ── Recherche automatique des meilleures analogies ───────────────────────────
print("[...] Recherche des meilleures analogies (quelques secondes)...")

MIN_ANALOGY_SCORE = 0.60  # similarité cosinus minimale pour retenir une analogie

# ── Directions sémantiques prédéfinies pour (A, B) ──────────────────────────
# La relation A→B doit avoir un sens (genre, parenté, domaine, etc.)
semantic_directions = [
    # genre
    ("homme",  "femme"),   ("femme",  "homme"),
    ("roi",    "reine"),   ("reine",  "roi"),
    ("fils",   "fille"),   ("fille",  "fils"),
    ("père",   "mère"),    ("mère",   "père"),
    ("dieux",  "déesse"),  ("déesse", "dieux"),
    ("frère",  "sœur"),
    # parenté/hiérarchie
    ("fils",   "père"),    ("père",   "fils"),
    ("mère",   "fils"),
    # vie/mort/conflit
    ("mort",   "vie"),     ("vie",    "mort"),
    ("guerre", "paix"),    ("paix",   "guerre"),
    # espace
    ("mer",    "terre"),   ("terre",  "mer"),
    # personnages de l'Odyssée
    ("odysseus", "ithakè"),
    ("odysseus", "subtil"),
    ("athènè",   "déesse"),
    # qualités épiques
    ("subtil", "prudent"), ("divin",  "illustre"),
]
# Ne garder que les paires où les deux mots existent dans le vocabulaire
semantic_directions = [
    (a, b) for a, b in semantic_directions if a in corpus_model.wv and b in corpus_model.wv
]

# ── Pool de mots de contenu pour C ──────────────────────────────────────────
extra_stopwords = {
    "car", "ainsi", "certes", "point", "après", "avant", "quand", "alors",
    "donc", "mais", "puis", "enfin", "bien", "très", "tout", "tous", "toute",
    "toutes", "même", "aussi", "encore", "déjà", "moins", "plus", "lors",
    "voilà", "voici", "cela", "ceci", "celui", "celle", "ceux", "celles",
    "leur", "leurs", "lequel", "laquelle", "lesquels", "dont",
    "tandis", "comme", "pendant", "depuis", "jusqu", "vers", "chez", "par",
    "pour", "sans", "sous", "sur", "entre", "dans", "avec", "près",
    "devant", "derrière", "auprès", "autour", "afin", "lorsque", "jamais",
    "toujours", "parfois", "souvent", "loin", "dessus", "dessous",
    "cette", "autre", "autres", "parmi", "haute", "maintenant", "aussitôt",
    "ailées", "paroles",  # expression formulaique homérique → trop dominante
    "clairs", "yeux",     # "yeux clairs" d'Athéna → trop formulaique
}

content_pool = [
    w for w in corpus_words[:600]
    if len(w) >= 4
    and w not in stopwords_fr
    and w not in extra_stopwords
][:150]
pool_set = set(content_pool)

word_to_idx_c = {w: i for i, w in enumerate(corpus_words)}
pool_indices = [word_to_idx_c[w] for w in content_pool]

# Vecteurs normalisés pour tout le vocabulaire
all_vecs_c = corpus_model.wv.vectors.copy()          # (vocab, dim)
all_norms_c = np.linalg.norm(all_vecs_c, axis=1, keepdims=True)
all_vecs_c_norm = all_vecs_c / (all_norms_c + 1e-9)  # (vocab, dim)

pool_vecs_norm = all_vecs_c_norm[pool_indices]          # (pool, dim) normalisés
pool_size = len(content_pool)

# ── Recherche vectorisée : pour chaque direction (A→B), meilleur C ──────────
# On utilise les vecteurs normalisés comme le fait gensim.wv.most_similar
good_analogies = []

for a, b in semantic_directions:
    idx_a = word_to_idx_c[a]
    idx_b = word_to_idx_c[b]
    direction = all_vecs_c_norm[idx_b] - all_vecs_c_norm[idx_a]   # (dim,)

    # Pour tous les C du pool : cible = norm(v_C) + direction
    targets = pool_vecs_norm + direction[np.newaxis, :]  # (pool, dim)
    t_norms = np.linalg.norm(targets, axis=1, keepdims=True)
    targets_norm = targets / (t_norms + 1e-9)
    # Similarités cosinus avec tout le vocabulaire
    sims = targets_norm @ all_vecs_c_norm.T              # (pool, vocab)
    # Exclure A et B de tous les résultats
    sims[:, idx_a] = -np.inf
    sims[:, idx_b] = -np.inf
    # Exclure chaque C de son propre résultat
    for k in range(pool_size):
        sims[k, pool_indices[k]] = -np.inf

    best_idxs = np.argmax(sims, axis=1)                 # (pool,)
    best_scores = sims[np.arange(pool_size), best_idxs]

    for k in range(pool_size):
        score = float(best_scores[k])
        c_word = content_pool[k]
        if c_word in {a, b}:      # C ne doit pas être A ni B
            continue
        if score >= MIN_ANALOGY_SCORE:
            d_word = corpus_words[int(best_idxs[k])]
            if d_word in pool_set and d_word not in {a, b, c_word}:
                good_analogies.append({
                    "a": a, "b": b, "c": c_word, "d": d_word, "score": score,
                })

# Tri et dédoublonnage
good_analogies.sort(key=lambda x: -x["score"])
seen_abc = set()
seen_d = set()
unique_analogies = []
for ana in good_analogies:
    a, b, c, d = ana["a"], ana["b"], ana["c"], ana["d"]
    key = (a, b, c)
    # Exclure variantes morphologiques (un mot est préfixe d'un autre)
    words4 = [a, b, c, d]
    morpho = any(
        w1 != w2 and (w1.startswith(w2) or w2.startswith(w1))
        for ii, w1 in enumerate(words4) for w2 in words4[ii+1:]
    )
    if key not in seen_abc and d not in seen_d and not morpho:
        seen_abc.add(key)
        seen_d.add(d)
        unique_analogies.append(ana)

print(f"[OK] {len(unique_analogies)} bonnes analogies trouvées (score >= {MIN_ANALOGY_SCORE})")
for ana in unique_analogies[:10]:
    print(f"  {ana['a']} : {ana['b']} :: {ana['c']} : {ana['d']} ({ana['score']:.3f})")

# Récupération des top-5 résultats complets pour l'affichage HTML
analogies_results = []
for ana in unique_analogies[:5]:
    try:
        full_res = corpus_model.wv.most_similar(
            positive=[ana["b"], ana["c"]], negative=[ana["a"]], topn=5
        )
        analogies_results.append({
            "a": ana["a"], "b": ana["b"], "c": ana["c"],
            "results": [(w, float(s)) for w, s in full_res],
        })
    except KeyError:
        pass

# Repli sur des triples prédéfinis si aucune bonne analogie n'est trouvée
if not analogies_results:
    print("[WARN] Aucune bonne analogie trouvée, repli sur les triples prédéfinis.")
    fallback = [
        ("homme", "roi", "femme"), ("fils", "père", "fille"),
        ("mort", "vie", "guerre"), ("dieux", "homme", "déesse"),
    ]
    for a, b, c in fallback:
        if all(w in corpus_model.wv for w in (a, b, c)):
            res = word_analogy(corpus_model, a, b, c, topn=5)
            if res:
                analogies_results.append({
                    "a": a, "b": b, "c": c,
                    "results": [(w, float(s)) for w, s in res],
                })
    analogies_results = analogies_results[:5]

# Top similar for 3 key words, 5 closest each
SIM_WORDS = ["odysseus", "mer", "dieux"]
# Fallback: if words not in vocab, use main_highlight
sim_word_list = [w for w in SIM_WORDS if w in corpus_model.wv]
if len(sim_word_list) < 3:
    sim_word_list = main_highlight[:3]
similarity_results = {}
for w in sim_word_list[:3]:
    res = top_similar(corpus_model, w, topn=5)
    if res:
        similarity_results[w] = [(rw, float(rs)) for rw, rs in res]

# Save
with open(os.path.join(OUTDIR, "corpus_analogies.json"), "w") as f:
    json.dump(analogies_results, f, ensure_ascii=False, indent=2)

with open(os.path.join(OUTDIR, "corpus_similarities.json"), "w") as f:
    json.dump(similarity_results, f, ensure_ascii=False, indent=2)

# ── Analogy PCA visualization SVG ───────────────────────────

# Pick interesting words for analogy viz
analogy_viz_words = set()
for ar in analogies_results:
    analogy_viz_words.add(ar["a"])
    analogy_viz_words.add(ar["b"])
    analogy_viz_words.add(ar["c"])
    for rw, _ in ar["results"][:3]:
        analogy_viz_words.add(rw)

# Also add similar words for main highlights
for w in main_highlight:
    analogy_viz_words.add(w)
    if w in similarity_results:
        for rw, _ in similarity_results[w][:5]:
            analogy_viz_words.add(rw)

analogy_viz_words = [w for w in analogy_viz_words if w in corpus_model.wv]
if len(analogy_viz_words) > 5:
    analogy_indices = [corpus_words.index(w) for w in analogy_viz_words]
    analogy_2d = corpus_final_2d[analogy_indices]

    W_av = 600
    H_av = 450
    center_av = (W_av / 2, H_av / 2)
    av_scale = 160.0 / max(1e-9, np.abs(analogy_2d).max())

    av_svg = svg_head(W_av, H_av)
    av_svg += f'<rect x="0" y="0" width="{W_av}" height="{H_av}" fill="#fcfcfc" rx="4"/>\n'
    av_svg += f'<line x1="15" y1="{center_av[1]}" x2="{W_av-15}" y2="{center_av[1]}" stroke="#ccc" stroke-width="0.4"/>\n'
    av_svg += f'<line x1="{center_av[0]}" y1="15" x2="{center_av[0]}" y2="{H_av-15}" stroke="#ccc" stroke-width="0.4"/>\n'

    # Compute dot positions and initial label positions
    dots = []
    for i, w in enumerate(analogy_viz_words):
        vx, vy = analogy_2d[i]
        sx = center_av[0] + vx * av_scale
        sy = center_av[1] - vy * av_scale
        is_main = w in main_highlight
        dx = sx - center_av[0]
        off = 7 if dx >= 0 else -7
        anchor = "start" if dx >= 0 else "end"
        # label bbox estimate: ~6px per char width, 12px height
        lw = len(w) * 6
        lx = sx + off
        ly = sy + 3
        dots.append({"w": w, "sx": sx, "sy": sy, "lx": lx, "ly": ly,
                      "lw": lw, "anchor": anchor, "is_main": is_main})

    # Greedy label nudging: push labels apart when bboxes overlap
    def label_bbox(d):
        if d["anchor"] == "start":
            x0, x1 = d["lx"], d["lx"] + d["lw"]
        else:
            x0, x1 = d["lx"] - d["lw"], d["lx"]
        y0, y1 = d["ly"] - 10, d["ly"] + 3
        return x0, y0, x1, y1

    def overlaps(a, b):
        ax0, ay0, ax1, ay1 = label_bbox(a)
        bx0, by0, bx1, by1 = label_bbox(b)
        return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0

    for _ in range(50):  # iterate until stable
        moved = False
        for i in range(len(dots)):
            for j in range(i + 1, len(dots)):
                if overlaps(dots[i], dots[j]):
                    # Push apart vertically
                    if dots[i]["ly"] <= dots[j]["ly"]:
                        dots[i]["ly"] -= 4
                        dots[j]["ly"] += 4
                    else:
                        dots[i]["ly"] += 4
                        dots[j]["ly"] -= 4
                    moved = True
        if not moved:
            break

    # Draw dots first, then labels
    for d in dots:
        col = "#286c44" if d["is_main"] else "#777"
        r = "3" if d["is_main"] else "2.5"
        av_svg += f'<circle cx="{d["sx"]:.1f}" cy="{d["sy"]:.1f}" r="{r}" fill="{col}" opacity="0.7"/>\n'
    for d in dots:
        col = "#286c44" if d["is_main"] else "#777"
        fw = "600" if d["is_main"] else "400"
        av_svg += f'<text x="{d["lx"]:.1f}" y="{d["ly"]:.1f}" fill="{col}" font-size="10" font-weight="{fw}" font-family="serif" text-anchor="{d["anchor"]}">{d["w"]}</text>\n'

    av_svg += svg_tail()
    write_svg(os.path.join(OUTDIR, "corpus_analogy_scatter.svg"), av_svg)

# ═══════════════════════════════════════════════════════════════
#  Inject dynamic content into index.html (no JavaScript needed)
# ═══════════════════════════════════════════════════════════════

HTML_PATH = "index.html"
with open(HTML_PATH, "r", encoding="utf-8") as f:
    html_content = f.read()

def inject(html_str, name, content):
    """Replace content between <!-- INJECT:name --> and <!-- /INJECT:name -->."""
    pattern = rf'(<!-- INJECT:{name} -->).*?(<!-- /INJECT:{name} -->)'
    return re.sub(pattern, rf'\1\n{content}\2', html_str, flags=re.DOTALL)

# 1. Similarity table rows
sim_rows = ""
for w1, w2, s in sim_table:
    sign = "+" if s >= 0 else ""
    bar_w = round(abs(s) * 80)
    bar_col = "#286c44" if s >= 0 else "#6aad8a"
    sim_rows += f'  <tr><td>{html.escape(w1)}</td><td>{html.escape(w2)}</td>'
    sim_rows += f'<td class="sv">{sign}{s:.3f}</td>'
    sim_rows += f'<td><span class="bar" style="width:{bar_w}px;background:{bar_col}"></span></td></tr>\n'
html_content = inject(html_content, "sim_table", sim_rows)

# 2. Corpus text excerpt
corpus_excerpt = html.escape(corpus_text[:8000]) + "\n\n[…]"
html_content = inject(html_content, "corpus_text", corpus_excerpt)

# 3. Top-similar words
sim_html = ""
for word, sims in similarity_results.items():
    sim_html += f'<div class="sim-item"><h4>{html.escape(word)}</h4><ol>'
    for rw, rs in sims[:5]:
        sim_html += f'<li>{html.escape(rw)} <span class="score">{rs:.3f}</span></li>'
    sim_html += '</ol></div>\n'
html_content = inject(html_content, "similarities", sim_html)

# 4. Analogy results
ana_html = ""
for ar in analogies_results:
    ana_html += '<div class="analogy">'
    ana_html += f'<span class="eq">{html.escape(ar["a"])}</span> est à '
    ana_html += f'<span class="eq">{html.escape(ar["b"])}</span> ce que '
    ana_html += f'<span class="eq">{html.escape(ar["c"])}</span> est à … '
    if ar["results"]:
        ana_html += f'<strong>{html.escape(ar["results"][0][0])}</strong>'
        ana_html += f' <span class="score">({ar["results"][0][1]:.3f})</span>'
    ana_html += '<div class="res">'
    for rw, rs in ar["results"][:5]:
        ana_html += f'{html.escape(rw)} ({rs:.3f}) · '
    ana_html += '</div></div>\n'
html_content = inject(html_content, "analogies", ana_html)

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html_content)
print("[OK] Injected dynamic content into index.html")

print("\n[DONE] All outputs written to", OUTDIR)
print("Files:", sorted(os.listdir(OUTDIR)))
