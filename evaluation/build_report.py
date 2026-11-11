#!/usr/bin/env python3
"""Render the face-similarity analysis as a standalone HTML report.

Reads only committed artifacts (results_invariance/, results_adaface/) plus the
saved aligned crops, and writes evaluation/report/ — an index.html with figures
and example images that opens directly in a browser with no server and no code.

    evaluation/.venv/bin/python evaluation/build_report.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

# Run as a script from anywhere: Python puts evaluation/ first on sys.path, so the
# local measurement module imports without path manipulation.
import face_invariance as fi

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUT = HERE / "report"
FIG = OUT / "figures"
IMG = OUT / "images"

def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(
            f"missing result file: {path}\nRun the measurement scripts first.") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"malformed JSON in {path}: {exc}") from exc


INV = load_json(HERE / "results_invariance/invariance.json")
AB = load_json(HERE / "results_adaface/ab_scores.json")
CURVES = list(csv.DictReader((HERE / "results_invariance/curves.csv").open()))
IT01 = list(csv.DictReader((HERE / "results_invariance/iteration01_check.csv").open()))

IDS = ("maya", "priya", "jonah")
COL = {"maya": "#1f77b4", "priya": "#e07b39", "jonah": "#c0392b"}
INK, MUTED, RULE = "#1c1c1e", "#6b6b70", "#d8d8dd"
ACCENT, WARN = "#0b5cad", "#b3261e"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 9,
    "axes.edgecolor": RULE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": "#ececf0", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
})

T = INV["floor"]["T"]
GAP = INV["between_identity"]


def row_drop(row: dict) -> float:
    """Worst single-identity cosine drop in one curve row."""
    vals = [1 - float(row[f"cosine_person_{i}"]) for i in IDS if row.get(f"cosine_person_{i}")]
    return max(vals) if vals else float("nan")


def embed_only_worst(subfamily: str) -> float:
    """Worst-identity drop for one sub-family with alignment removed, realistic tier."""
    return max(row_drop(r) for r in CURVES
               if r["regime"] == "embed_only" and r["subfamily"] == subfamily
               and r["realistic"] == "True")


def level_num(level: str) -> int:
    return int(level.split("_")[-1].rstrip("pxK"))


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

FAMILIES = [
    ("illumination", "gamma", "gamma"),
    ("illumination", "shading", "shadow gradient"),
    ("resolution", "jpeg", "JPEG q50"),
    ("resolution", "blur", "blur"),
    ("illumination", "gain_offset", "brightness"),
    ("illumination", "temperature", "colour cast"),
    ("pose", "roll", "roll (in plane)"),
    ("pose", "yaw", "yaw (turn)"),
    ("occlusion", "mouth", "mouth covered"),
    ("pose", "pitch", "pitch (nod)"),
    ("resolution", "downscale", "low resolution"),
    ("occlusion", "eyes", "eyes covered"),
]


def fig_sensitivity() -> None:
    worst = {}
    for r in CURVES:
        if r["realistic"] != "True" or r["family"] == "none":
            continue
        key = (r["regime"], r["subfamily"])
        d = row_drop(r)
        if key not in worst or d > worst[key]["d"]:
            worst[key] = {"d": d, "level": r["level"]}

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    y = np.arange(len(FAMILIES))
    e = [worst.get(("end_to_end", s), {}).get("d", np.nan) for _, s, _ in FAMILIES]
    b = [worst.get(("embed_only", s), {}).get("d", np.nan) for _, s, _ in FAMILIES]
    h = 0.38
    ax.barh(y + h / 2, e, h, label="full pipeline", color=ACCENT)
    ax.barh(y - h / 2, b, h, label="embedder alone", color="#8ab4e8")
    for yy, dark, light in zip(y.tolist(), e, b, strict=True):
        ax.text(np.nanmax([dark, light]) + 0.008, float(yy), f"{dark:.2f}",
                va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, _, label in FAMILIES])
    ax.invert_yaxis()
    ax.set_xlabel("worst-identity drop in cosine (identity unchanged)")
    ax.set_xlim(0, 0.42)
    ax.axvline(T, color=WARN, ls="--", lw=1.1)
    ax.text(T + 0.005, 1.004, f"T = {T:.2f}", color=WARN, fontsize=8, va="bottom",
            transform=ax.get_xaxis_transform())
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.10))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def fig_pose() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), sharey=True)
    for ax, regime, title in zip(axes, ("end_to_end", "embed_only"),
                                 ("full pipeline (detect → align → embed)",
                                  "embedder alone (no re-alignment)"), strict=True):
        for sub, color in (("roll", "#2e7d32"), ("yaw", ACCENT), ("pitch", WARN)):
            pts = {}
            for r in CURVES:
                if r["regime"] == regime and r["family"] == "pose" and r["subfamily"] == sub:
                    deg = abs(int(r["level"].split(sub)[1]))
                    pts[deg] = max(pts.get(deg, 0.0), row_drop(r))
            xs = sorted(pts)
            ax.plot(xs, [pts[x] for x in xs], "o-", ms=4, lw=1.6, color=color, label=sub)
        ax.axvspan(0, 20, color="#2e7d32", alpha=0.07)
        ax.set_xlabel("rotated by (degrees)")
        ax.set_title(title, fontsize=9, color=MUTED)
    axes[0].set_ylabel("worst-identity drop")
    axes[0].annotate("ordinary\nphoto range", (10, 0.42), ha="center", fontsize=7.5,
                     color="#2e7d32", style="italic")
    axes[0].legend(frameon=False, fontsize=8, title="rotation", title_fontsize=8)
    fig.suptitle("Pitch costs far more than yaw, and the aligner hides in-plane roll",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "pose.png", bbox_inches="tight")
    plt.close(fig)


def fig_resolution() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for ident in IDS:
        pts = {}
        for r in CURVES:
            if r["regime"] == "end_to_end" and r["subfamily"] == "downscale":
                v = r[f"cosine_person_{ident}"]
                if v:
                    pts[level_num(r["level"])] = float(v)
        xs = sorted(pts)
        label = f"{ident} (smallest face)" if ident == "jonah" else ident
        ax.plot(xs, [pts[x] for x in xs], "o-", ms=4, lw=1.6, color=COL[ident], label=label)
    ax.axhline(0, color=INK, lw=1)
    ax.axvspan(160, 1100, color="#2e7d32", alpha=0.07)
    ax.set_xscale("log")
    ax.set_xticks([40, 56, 80, 112, 160, 224])
    ax.set_xticklabels(["40", "56", "80", "112", "160", "224"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(36, 260)
    ax.set_ylim(-0.16, 1.04)
    ax.set_xlabel("resolution of the face, measured in pixels across")
    ax.set_ylabel("cosine to the same person's reference")
    ax.text(58, -0.135, "at 40 px a genuine match falls to −0.02,\n"
                        "below an unrelated person",
            fontsize=7.5, color=WARN, va="bottom")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "resolution.png", bbox_inches="tight")
    plt.close(fig)


def fig_envelope() -> None:
    """The headline: nuisance noise versus the signal we actually want to measure."""
    swap = GAP["identity_swap_drop_hardest"]
    impostor = GAP["max_cross_identity_cosine"]
    share = GAP["nuisance_share"]

    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(7.6, 3.4), gridspec_kw={"height_ratios": [2.7, 1]})

    # Panel A: where the noise sits on the cosine scale.
    ax.axhline(0, color=RULE, lw=3, solid_capstyle="butt")
    ax.axvspan(1 - T, 1.0, color=WARN, alpha=0.10)
    ax.plot([1.0], [0.0], "o", ms=9, color=INK, zorder=4)
    ax.plot([float(impostor)], [0.0], "o", ms=9, color="#2e7d32", zorder=4)
    ax.text(0.99, 0.10, "same portrait\ncosine = 1.00", ha="right", va="bottom",
            fontsize=8, color=INK)
    ax.text(float(impostor) - 0.03, 0.10, f"a different person\nhardest pair = {impostor:.2f}",
            ha="right", va="bottom", fontsize=8, color="#2e7d32")
    ax.text(1.0, -0.20, f"T = {T:.2f} — a change this small could be\n"
                        "lighting, pose, blur or occlusion",
            ha="right", va="top", fontsize=8, color=WARN)
    ax.text(float(impostor) + (1 - T - float(impostor)) / 2, -0.20,
            f"usable signal: {swap - T:.2f}", ha="center", va="top",
            fontsize=8, color=MUTED)
    ax.set_xlim(-0.01, 1.03)
    ax.set_ylim(-0.75, 0.45)
    ax.set_yticks([])
    ax.set_xlabel("cosine similarity")
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_title("How much of a real identity change can the metric actually see?",
                 fontsize=10, pad=8)

    # Panel B: the same ratio as a share of one swap. Its own scale — deliberately
    # not shared with the cosine axis above.
    axb.barh(0, share, 0.5, color=WARN, alpha=0.8)
    axb.barh(0, 1 - share, 0.5, left=share, color=ACCENT, alpha=0.9)
    axb.text(share / 2, 0, f"{share:.0%} lost to nuisance", ha="center", va="center",
             color="white", fontsize=8.5, weight="bold")
    axb.text(share + (1 - share) / 2, 0, f"{1 - share:.0%} usable",
             ha="center", va="center", color="white", fontsize=8.5, weight="bold")
    axb.set_yticks([])
    axb.set_xlim(0, 1)
    axb.set_xticks([0, share, 1.0])
    axb.set_xticklabels(["0%", f"{share:.0%}", "100%"])
    axb.set_xlabel(f"one full identity swap ({swap:.2f}), shared out")
    axb.grid(False)
    axb.spines["left"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG / "envelope.png", bbox_inches="tight")
    plt.close(fig)


def fig_adaface() -> None:
    pairs = AB["comparison"]["scored_pairs"]
    a = np.array([p["arcface_similarity"] for p in pairs])
    b = np.array([p["adaface_similarity"] for p in pairs])
    fig, ax = plt.subplots(figsize=(4.4, 4.1))
    lim = (float(min(a.min(), b.min())) - 0.05, float(max(a.max(), b.max())) + 0.05)
    ax.plot(lim, lim, "--", color=MUTED, lw=1, label="identical score")
    ax.axhline(0, color=RULE, lw=0.9)
    ax.axvline(0, color=RULE, lw=0.9)
    ax.scatter(a, b, s=36, color=ACCENT, zorder=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("ArcFace cosine")
    ax.set_ylabel("AdaFace cosine")
    ax.set_title(f"{len(pairs)} scored appearances  (ρ = "
                 f"{AB['comparison']['spearman_assigned_similarities']:.2f})", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "adaface.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Example images: what each nuisance actually does to the face the model sees
# --------------------------------------------------------------------------

ENTRIES = {e["level"]: e for e in fi.perturbation_grid()}

NICE = {
    "yaw+20": "yaw +20°", "yaw-20": "yaw −20°", "pitch+20": "pitch +20°",
    "pitch-20": "pitch −20°", "pitch-40": "pitch −40°", "roll+20": "roll +20°",
    "gamma_0.7": "dimmed", "gamma_1.4": "brightened",
    "temperature_3000K": "warm light", "temperature_9000K": "cool light",
    "shading_0.5_at_90deg": "side light",
    "eyes_5pct": "eyes 5%", "eyes_10pct": "eyes 10%", "eyes_20pct": "eyes 20%",
    "mouth_10pct": "mouth 10%", "mask_band": "mask", "sunglasses_band": "sunglasses",
    "downscale_160px": "160 px", "downscale_112px": "112 px",
    "downscale_80px": "80 px", "downscale_40px": "40 px",
}


def maya_crop_and_focal() -> tuple[np.ndarray, float]:
    ref = next(r for r in INV["references"] if r["identity"].endswith("maya"))
    portrait = cv2.imread(str(PROJECT / ref["source"]))
    if portrait is None:
        raise SystemExit(f"missing portrait: {ref['source']}")
    crop = cv2.imread(str(PROJECT / ref["aligned_crop"]))
    if crop is None:
        raise SystemExit(f"missing aligned crop: {ref['aligned_crop']}")
    face_width = float(ref["bbox"][2] - ref["bbox"][0])
    focal = fi.FOCAL_PER_FRAME_WIDTH * portrait.shape[1] * fi.ALIGNED_CROP_SIZE / face_width
    return crop, focal


def strip(name: str, levels: list[str], note: str = "") -> None:
    crop, focal = maya_crop_and_focal()
    centre = tuple(fi.ALIGNED_TEMPLATE_KPS.mean(axis=0))
    tiles, labels = [], []
    for level in levels:
        if level == "original":
            tiles.append(crop)
            labels.append("original")
            continue
        entry = ENTRIES[level]
        out = fi.apply_perturbation(crop, entry, kps=fi.ALIGNED_TEMPLATE_KPS,
                                    focal_px=focal, centre=centre, seed=fi.BASE_SEED)
        tiles.append(fi.as_uint8(out))
        labels.append(NICE.get(level, level))

    fig, axes = plt.subplots(1, len(tiles), figsize=(1.45 * len(tiles), 1.78))
    for ax, tile, label in zip(np.atleast_1d(axes), tiles, labels, strict=True):
        ax.imshow(tile)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(label, fontsize=8, color=INK, pad=4)
    fig.text(0.5, -0.04, note, ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def copy_references() -> None:
    for ref in INV["references"]:
        path = PROJECT / ref["source"]
        portrait = cv2.imread(str(path))
        if portrait is None:
            raise SystemExit(f"cannot read portrait: {path}")
        name = ref["identity"].replace("person_", "")
        height, width = portrait.shape[:2]
        scale = 320 / height
        small = cv2.resize(portrait, (int(width * scale), 320),
                           interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(IMG / f"ref_{name}.jpg"), small,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

CSS = """
:root { --ink:#1c1c1e; --muted:#6b6b70; --rule:#e3e3e8; --accent:#0b5cad;
        --warn:#b3261e; --bg:#ffffff; --panel:#f7f7f9; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif;
       -webkit-font-smoothing:antialiased; }
main { max-width:820px; margin:0 auto; padding:56px 24px 96px; }
h1 { font-size:2rem; line-height:1.2; letter-spacing:-0.02em; margin:0 0 12px; }
h2 { font-size:1.22rem; letter-spacing:-0.01em; margin:56px 0 14px;
     padding-top:22px; border-top:1px solid var(--rule); }
h3 { font-size:1rem; margin:28px 0 8px; }
p { margin:0 0 15px; }
a { color:var(--accent); text-decoration:none; }
a:hover { text-decoration:underline; }
.lede { font-size:1.09rem; color:#33333a; }
.sub { color:var(--muted); font-size:0.94rem; margin:0 0 32px; }
figure { margin:26px 0; }
figure img { width:100%; display:block; border-radius:4px; }
figcaption { font-size:0.86rem; color:var(--muted); margin-top:10px; line-height:1.5; }
.posetile { border:1px solid var(--rule); }
.box { background:var(--panel); border-radius:6px; padding:18px 20px; margin:24px 0; }
.box.warn { background:#fdf3f2; border-left:3px solid var(--warn); }
.box p:last-child, .box ul:last-child { margin-bottom:0; }
.box h3 { margin-top:0; }
dl.key { display:grid; grid-template-columns:auto 1fr; gap:6px 18px; margin:0; }
dl.key dt { font-weight:600; font-variant-numeric:tabular-nums; white-space:nowrap; }
dl.key dd { margin:0; color:var(--muted); }
table { border-collapse:collapse; width:100%; font-size:0.88rem; margin:16px 0;
        font-variant-numeric:tabular-nums; }
th { text-align:left; font-weight:600; border-bottom:1.5px solid var(--ink);
     padding:8px 10px 8px 0; }
td { border-bottom:1px solid var(--rule); padding:7px 10px 7px 0; }
th.num, td.num { text-align:right; }
tr.hi td { background:#fdf3f2; }
tr.tie td { background:#fffbe8; }
code { font:0.88em ui-monospace,SFMono-Regular,Menlo,monospace;
       background:var(--panel); padding:1px 5px; border-radius:3px; }
.gallery { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:22px 0; }
.gallery figure { margin:0; }
.gallery figcaption { font-size:0.8rem; margin-top:6px; }
.pipe { display:flex; align-items:center; gap:14px; margin:24px 0; flex-wrap:wrap; }
.pipe figure { margin:0; text-align:center; }
.pipe img { width:118px; border-radius:4px; }
.pipe .arrow { color:var(--muted); font-size:1.3rem; }
.pipe figcaption { font-size:0.78rem; margin-top:6px; max-width:130px; }
footer { margin-top:64px; padding-top:22px; border-top:1px solid var(--rule);
         font-size:0.82rem; color:var(--muted); }
@media (max-width:620px) { .gallery { grid-template-columns:repeat(2,1fr); }
                           dl.key { grid-template-columns:1fr; gap:2px; }
                           dl.key dd { margin-bottom:8px; } }
"""


def fmt(x: float, n: int = 2) -> str:
    return f"{x:.{n}f}"


def build_html() -> str:
    worst = dict(INV["realistic_family_maxima"])

    # --- sensitivity table
    order = ["illumination/gamma", "illumination/shading", "illumination/gain_offset",
             "resolution/jpeg", "resolution/blur", "illumination/temperature",
             "pose/roll", "pose/yaw", "occlusion/mouth", "resolution/downscale",
             "pose/pitch", "occlusion/eyes"]
    sens_rows = []
    for key in order:
        v = worst[key]
        fam, sub = key.split("/")
        cls = ' class="hi"' if v["max_drop"] >= 0.23 else ""
        sens_rows.append(
            f'<tr{cls}><td>{sub.replace("_", " ")}</td><td>{fam}</td>'
            f'<td class="num">{fmt(v["max_drop"])}</td>'
            f'<td class="num">{fmt(v["mean_drop"])}</td></tr>')

    # --- iteration01 table
    it_rows = []
    for r in IT01:
        if r["condition"] != "historical":
            continue
        drop = float(r["drop"])
        if r["exceeds_T"] == "True":
            verdict, cls = "tie with T", ' class="tie"'
        else:
            verdict, cls = "inside the noise", ""
        it_rows.append(
            f'<tr{cls}><td>month {r["month"]}</td><td class="num">{fmt(float(r["pass1"]), 3)}</td>'
            f'<td class="num">{fmt(float(r["pass2"]), 3)}</td>'
            f'<td class="num">{drop:+.4f}</td><td>{verdict}</td></tr>')

    ctrl = {r["month"]: r for r in IT01 if r["condition"] == "control"}

    # --- adaface
    pa = AB["comparison"]["per_arm"]
    dl = AB["comparison"]["similarity_delta_adaface_minus_arcface"]
    chg = AB["comparison"]["assignment_changes"]
    chg_rows = "".join(
        f'<tr><td>month {c["month"]}, pass {c["pass_index"] + 1}</td>'
        f'<td>{c["person_id"].replace("person_", "")}</td>'
        f'<td>{"unassigned" if c["arcface_face_index"] is None else "face " + str(c["arcface_face_index"])}</td>'
        f'<td>{"unassigned" if c["adaface_face_index"] is None else "face " + str(c["adaface_face_index"])}</td></tr>'
        for c in chg)

    # --- maya pass1 -> pass2 comparison for both arms
    maya = {}
    for t in AB["comparison"]["pass_to_pass_transitions"]:
        if t["person_id"].endswith("maya"):
            maya.setdefault(t["arm"], {})[t["month"]] = t["series"]
    maya_rows = []
    for month in (1, 2, 8, 10):
        cells = []
        for arm in ("arcface", "adaface"):
            s = maya[arm][month]
            v = [x["similarity"] for x in s]
            if v[1] is None:
                cells.append(f'<td class="num">{fmt(v[0], 3)} → dropped</td>')
            else:
                cells.append(f'<td class="num">{fmt(v[0], 3)} → {fmt(v[1], 3)}</td>')
        maya_rows.append(f'<tr><td>month {month}</td>{"".join(cells)}</tr>')

    sheet_note = (
        "Example images are rendered from the saved aligned 112 × 112 crop of Maya's "
        "canonical portrait by the same perturbation code that produced the measurements, "
        "so they show exactly what the face recogniser was fed. Tiles are shown at native "
        "size; pose tiles differ slightly in shape because a perspective warp changes the "
        "frame. Contact sheets for every level and all three characters are regenerated by "
        "<code>evaluation/face_invariance.py</code>."
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>How reliable is the face-similarity score?</title>
<style>{CSS}</style>
</head>
<body>
<main>

<h1>How reliable is the face-similarity score?</h1>
<p class="sub">An analysis of the AI Camera Roll face-consistency metric. Static page —
no server, no code, no network. Numbers come from the committed result files in
<code>evaluation/results_invariance/</code> and <code>evaluation/results_adaface/</code>.</p>

<p class="lede">The project measures whether a generated photo still shows the same fictional
person by comparing face embeddings and reporting a cosine similarity. Those numbers were
being read as if they were exact. They are not. This page quantifies how much the score
moves when the identity is <em>provably</em> unchanged — and then asks whether the claimed
effects in the project are larger than that movement.</p>

<div class="box">
<h3>The short version</h3>
<dl class="key">
<dt>{fmt(T)}</dt><dd>how far the score drifts from lighting, pose, blur and occlusion
alone — with identity untouched</dd>
<dt>{fmt(GAP['identity_swap_drop_hardest'])}</dt><dd>how far the score moves when the
person is genuinely swapped for someone else</dd>
<dt>{GAP['nuisance_share']:.0%}</dt><dd>of the available signal is therefore noise:
nuisance eats it before any real change is measured</dd>
<dt>0.05 – 0.30</dt><dd>the reported pass-to-pass drops — every one inside the noise, and
the largest clears it by 0.0006</dd>
<dt>No conclusion changed</dt><dd>swapping in a quality-aware face model (AdaFace): scores
shifted slightly and three pairings moved, but nothing flipped</dd>
</dl>
</div>

<h2>1. Why a raw similarity score needs a baseline</h2>

<p>The pipeline takes a canonical portrait of each character, finds faces in a generated
photo, and compares them with a face-recognition model (ArcFace). The output is a cosine
similarity between two 512-dimensional vectors: 1.0 means "the model considers these the
same face", values near 0 mean "unrelated".</p>

<div class="pipe">
  <figure><img src="images/ref_maya.jpg" alt="canonical portrait of Maya">
    <figcaption>canonical portrait</figcaption></figure>
  <span class="arrow">→</span>
  <figure><img src="images/aligned_maya.png" alt="aligned 112 pixel face crop"
    style="width:112px;image-rendering:auto">
    <figcaption>face located &amp; aligned to 112 × 112</figcaption></figure>
  <span class="arrow">→</span>
  <figure style="text-align:left"><img src="images/aligned_priya.png" alt="Priya's crop"
    style="width:112px">
    <figcaption>compared with each reference → one number</figcaption></figure>
</div>

<p>A score of 0.31 sounds informative. But informative <em>relative to what?</em> A face
recognition model is a measuring instrument, and nobody had characterised this
instrument's precision. Without that, "the score fell from 0.31 to 0.15" could mean the
identity degraded, or simply that the second image was turned, dimmer, or softer. The
two explanations are indistinguishable from the number alone.</p>

<p>So the first experiment does not touch the generator at all. It takes the three
canonical portraits, deliberately degrades copies of them in ways that <strong>cannot</strong>
change who the person is — rotate, relight, blur, shrink, cover part of the face — and
measures how far the score falls. Every point of that fall is pure measurement error.</p>

<h2>2. What was varied, and what counts as "realistic"</h2>

<p>Each portrait is pushed through two paths in parallel, because they answer different
questions:</p>

<table>
<tr><th>Path</th><th>What it includes</th><th>Why</th></tr>
<tr><td><strong>Full pipeline</strong></td>
    <td>detect the face again, re-align it, then embed</td>
    <td>this is what the project actually does, so it is the number that applies</td></tr>
<tr><td><strong>Embedder alone</strong></td>
    <td>perturb the cropped face, then embed</td>
    <td>separates the model's sensitivity from the alignment step's</td></tr>
</table>

<p>Only "realistic" degradations — the kind that still leave a human able to name the
person — are allowed to define the noise floor. The frozen rule: rotation within ±20°,
gamma 0.7–1.4, brightness 0.8–1.25×, three colour temperatures, at least 160 px across the
face, mild blur, JPEG quality ≥ 50, and occlusion covering at most 10 % of the face box.
Harsher levels are measured and reported, but they do not set the floor — otherwise the
noise estimate inflates until no real change could ever be detected.</p>

<h3>The degradations, on the face the model actually sees</h3>

<figure>
  <img class="posetile" src="figures/strip_pose.png" alt="Maya's face rotated">
  <figcaption><strong>Rotation.</strong> A perspective warp of the image. Note the frame
  changes shape — a real 3D turn would also hide the far cheek, which this does not, so
  the measured pose cost is an <em>under</em>-estimate.</figcaption>
</figure>

<figure>
  <img class="posetile" src="figures/strip_illumination.png" alt="Maya's face under different lighting">
  <figcaption><strong>Lighting.</strong> Dimming, brightening, a warm cast, a cool cast,
  and a directional shadow.</figcaption>
</figure>

<figure>
  <img class="posetile" src="figures/strip_occlusion.png" alt="Maya's face partly covered">
  <figcaption><strong>Covering part of the face.</strong> Grey rectangles over the eyes and
  mouth, plus typical sunglasses and mask bands.</figcaption>
</figure>

<figure>
  <img class="posetile" src="figures/strip_resolution.png" alt="Maya's face at lower resolutions">
  <figcaption><strong>Resolution.</strong> The face shrunk and restored, as happens when a
  character stands further from the camera in a generated scene.</figcaption>
</figure>

<p class="sub">{sheet_note}</p>

<h2>3. What the instrument actually does</h2>

<figure>
  <img src="figures/sensitivity.png" alt="Bar chart of worst-case score drop per degradation">
  <figcaption>Worst-case drop in cosine for each degradation, within the realistic range.
  Dark bars: the full pipeline. Light bars: the embedder with alignment already done.</figcaption>
</figure>

<table>
<tr><th>Degradation</th><th>Family</th><th class="num">Worst drop</th><th class="num">Typical drop</th></tr>
{''.join(sens_rows)}
</table>

<p>Three findings matter, and two of them are counter-intuitive:</p>

<ul>
<li><strong>Lighting is essentially free.</strong> Every brightness and colour change inside
the realistic range costs at most {fmt(worst['illumination/temperature']['max_drop'])}. Any
effort spent on illumination normalisation would be wasted — the model already handles it.</li>
<li><strong>Pose is not free, and it is lopsided.</strong> Turning the head sideways (yaw)
costs {fmt(worst['pose/yaw']['max_drop'])}; nodding it up or down (pitch) costs
{fmt(worst['pose/pitch']['max_drop'])}, roughly two and a half times as much. Tilting within
the picture plane looks almost free in the full pipeline
({fmt(worst['pose/roll']['max_drop'])}) because the alignment step undoes it — but take that
step away and the same tilt costs {fmt(embed_only_worst('roll'))}, more than ten times as
much. That gap is the alignment doing real work.</li>
<li><strong>Covering the eyes and losing resolution are the worst cases</strong> —
{fmt(worst['occlusion/eyes']['max_drop'])} and {fmt(worst['resolution/downscale']['max_drop'])}
— and neither is something the face-recognition literature would lead you to expect from a
modern embedding. A small grey patch over both eyes is visually trivial to a human and
expensive to the model.</li>
</ul>

<figure>
  <img src="figures/pose.png" alt="Score drop versus rotation angle">
  <figcaption>Drop against rotation angle, for each axis. The shaded band is the ordinary
  photographic range. In the left panel the aligner cancels in-plane roll; in the right
  panel, with alignment removed, roll becomes the most expensive of the three.</figcaption>
</figure>

<h3>Resolution: the failure is not uniform</h3>

<figure>
  <img src="figures/resolution.png" alt="Cosine versus face resolution, per character">
  <figcaption>Same person, decreasing resolution. Jonah's face is the smallest in the pilot
  (91 × 118 px versus 145 × 182 for Maya), and at 40 px his genuine match scores
  <em>below zero</em> — worse than an unrelated person.</figcaption>
</figure>

<p>This is the most awkward result in the analysis. A same-identity comparison cannot
legitimately score below a different-person comparison, yet at extreme resolution it does.
Practically it means resolution sensitivity is tied to how large a character happens to be
drawn: a character who is small in frame is judged more harshly for reasons that have
nothing to do with identity. Any comparison across characters is confounded by this.</p>

<h2>4. Turning that into a single number</h2>

<p>The noise floor <code>T</code> is the largest drop any one identity showed under a
realistic degradation: <strong>{fmt(T, 4)}</strong>. It is a bound, not an average — a
statement that no realistic nuisance in the tested grid moved any portrait further than
this while the identity stayed fixed.</p>

<figure>
  <img src="figures/envelope.png" alt="Nuisance noise versus identity-swap signal">
  <figcaption>Top: where the noise sits on the cosine scale, between "same portrait" and
  "different person". Bottom: the same ratio expressed as a share of one full identity
  swap.</figcaption>
</figure>

<p>Put against the size of a genuine identity change, the noise is large. The hardest
impostor pair in the reference gallery (Maya and Priya) sits at
{fmt(GAP['max_cross_identity_cosine'])}, so swapping one character for another moves the
score by {fmt(GAP['identity_swap_drop_hardest'])}. Nuisance alone accounts for
<strong>{GAP['nuisance_share']:.0%}</strong> of that. In signal-to-noise terms, the ratio is
{GAP['ratio']:.1f}.</p>

<p>The practical rule: <strong>a change smaller than about {fmt(T)} means nothing.</strong>
It is not evidence of improvement, and it is not evidence of regression.</p>

<h2>5. So do the project's claimed effects survive?</h2>

<p>The clearest claim in the project is that Maya's similarity fell after the second
generation pass in all four multi-pass scenes. Checking each drop against the floor:</p>

<table>
<tr><th>Scene</th><th class="num">Pass 1</th><th class="num">Pass 2</th><th class="num">Drop</th><th>Verdict</th></tr>
{''.join(it_rows)}
</table>

<p><strong>The direction reproduces; the size does not support the conclusion.</strong> The
January scene is the only one that comes close: it clears the floor by 0.0006, which is a tie,
not a separation. February (0.25), August (0.12) and October (0.05) are all inside it. In the
fresh control run, October's score actually <em>rose</em> slightly
({fmt(float(ctrl['10']['drop']), 4)}), which is what noise looks like.</p>

<p>None of this says the generator preserves identity well. It says the evidence offered for
identity loss is not strong enough to distinguish from lighting, pose and framing. Absence of
evidence, not evidence of absence.</p>

<div class="box warn">
<h3>An error to avoid repeating</h3>
<p>The first version of this analysis also flagged all 22 scene faces as "outside the
envelope" by comparing each score against 1.0. <strong>That comparison is invalid and
should not be quoted.</strong></p>
<p>The floor was measured on <em>a portrait versus a degraded copy of that same portrait</em>,
whose baseline is 1.0 by construction. A generated scene face is a different rendering
entirely and was never going to score near 1.0, so an absolute gate rejects every face
regardless of identity and carries no information. <code>T</code> is only valid as a budget
on the <strong>difference between two conditions</strong> — which is what the table above
does.</p>
<p>Fixing the absolute question needs a matched baseline: compare two independently
generated images of the <em>same</em> character. The aligned crops are already saved, so this
is cheap, and it is the top item in
<a href="../FUTURE_WORK.md">the backlog</a>.</p>
</div>

<h2>6. Would a better face model fix it?</h2>

<p>The obvious response to "the model is sensitive" is to use a better model. AdaFace
(CVPR 2022) is the natural candidate: it is trained with a quality-adaptive margin
specifically so that low-quality faces are handled better. The test swaps it in while
holding everything else fixed — the same detections, the same aligned crops, the same
references, the same pairing rule — so the model is the only variable.</p>

<figure>
  <img src="figures/adaface.png" alt="ArcFace versus AdaFace cosine for each appearance">
  <figcaption>Each point is one character appearance scored by both models. Points on the
  dashed line mean the two models agree exactly.</figcaption>
</figure>

<table>
<tr><th></th><th class="num">ArcFace</th><th class="num">AdaFace</th></tr>
<tr><td>Appearances assigned to someone</td>
    <td class="num">{pa['arcface']['assigned_count']}</td>
    <td class="num">{pa['adaface']['assigned_count']}</td></tr>
<tr><td>Near-ties (margin below 0.07)</td>
    <td class="num">{pa['arcface']['rank1_margins_below_band']} of {pa['arcface']['assigned_count']}</td>
    <td class="num">{pa['adaface']['rank1_margins_below_band']} of {pa['adaface']['assigned_count']}</td></tr>
<tr><td>Confident-looking but actually wrong-rank cases</td>
    <td class="num">{len(pa['arcface']['negative_rank1_margins'])}</td>
    <td class="num">{len(pa['adaface']['negative_rank1_margins'])}</td></tr>
<tr><td>Mean change in similarity (AdaFace − ArcFace)</td>
    <td class="num" colspan="2">{dl['mean']:+.3f}
    (range {dl['min']:+.3f} to {dl['max']:+.3f})</td></tr>
</table>

<p>The two models agree closely — rank correlation {AB['comparison']['spearman_assigned_similarities']:.2f}
on assigned pairs — but AdaFace scores slightly lower on average and produces <em>more</em>
near-ties, not fewer. Three pairings changed, all in the same October scene:</p>

<table>
<tr><th>Scene</th><th>Character</th><th>ArcFace</th><th>AdaFace</th></tr>
{chg_rows}
</table>

<p>And the headline claim does not flip. Maya's pass-to-pass numbers under both models:</p>

<table>
<tr><th>Scene</th><th class="num">ArcFace</th><th class="num">AdaFace</th></tr>
{''.join(maya_rows)}
</table>

<p><strong>Verdict: a lateral move, not a fix.</strong> A quality-aware model cannot rescue a
metric whose noise already consumes 40 % of its signal — the bottleneck is the measurement
design, not the model. The one genuinely useful outcome is a stability check: the two models
agree about Maya and disagree about the supporting characters, which suggests Maya's drift is
real while the others' is not. That kind of cross-model agreement is a better use of a second
model than replacing the first.</p>

<h2>7. What to do next</h2>

<p>Ranked, with effort and rationale, in <a href="../FUTURE_WORK.md">FUTURE_WORK.md</a>. The
three that follow directly from this page:</p>

<ol>
<li><strong>Build a matched baseline.</strong> Compare two generated images of the same
character, using the crops already saved. This replaces the invalid absolute gate and turns
the score into something with a real reference point.</li>
<li><strong>Test degradations in combination.</strong> The floor above varies one thing at a
time, which makes it a lower bound; real scenes combine pose, occlusion and resolution.</li>
<li><strong>Gate on quality.</strong> Report a quality estimate beside every similarity
instead of averaging unwinnable comparisons into a mean, so coverage is visible.</li>
</ol>

<h2>8. Limits of what this page establishes</h2>

<ul>
<li><strong>Three synthetic identities.</strong> Every number here rests on three portraits.
Expression, genuine 3D rotation, ageing and true profile views are untested, which makes
<code>T</code> a lower bound and every "inside the envelope" verdict permissive.</li>
<li><strong>Planar pose.</strong> The rotation is a 2D warp: no self-occlusion, unchanged
shading, and the detector still finds landmarks at extreme angles. Real pose sensitivity is
worse.</li>
<li><strong>One degradation at a time.</strong> See above.</li>
<li><strong>A tightly packed gallery.</strong> The references sit close together in
embedding space (up to {fmt(GAP['max_cross_identity_cosine'])} between different people), so a
"correct" pairing can be a near-tie — {pa['arcface']['rank1_margins_below_band']} of
{pa['arcface']['assigned_count']} are.</li>
<li><strong>The score is not a probability.</strong> There is no calibrated identity
threshold, and nothing here establishes one. This measures the sensitivity of a similarity,
not human-perceived identity, and it says nothing about image quality.</li>
<li><strong>Ground truth is a prompt, not a labelled image.</strong> A missing detection is
recorded as missing, never as zero.</li>
</ul>

<footer>
<p>Generated by <code>evaluation/build_report.py</code> from the committed result files.
Face recognition: ArcFace ResNet50 @ WebFace600K and AdaFace IR-101 @ WebFace12M, both run
on CPU via InsightFace and ONNX Runtime. Pretrained weights are research-use only and are
not redistributed here; checksums and sources are recorded in
<code>evaluation/results_invariance/invariance.json</code> and
<code>evaluation/adaface-provenance.json</code>.</p>
<p>Supporting detail: <a href="../FACE_INVARIANCE.md">FACE_INVARIANCE.md</a> (method),
<a href="../ADAFACE_AB.md">ADAFACE_AB.md</a> (model swap),
<a href="../face_metric_results.ipynb">the notebook</a> (same numbers, with code),
<a href="../FUTURE_WORK.md">FUTURE_WORK.md</a> (backlog). These open as plain text from
the local file system; view them on GitHub for the rendered form.</p>
</footer>

</main>
</body>
</html>
"""


def main() -> None:
    for d in (OUT, FIG, IMG):
        d.mkdir(parents=True, exist_ok=True)

    fig_sensitivity()
    fig_pose()
    fig_resolution()
    fig_envelope()
    fig_adaface()

    strip("strip_pose.png", ["original", "yaw+20", "pitch+20", "roll+20", "pitch-40"])
    strip("strip_illumination.png", ["original", "gamma_0.7", "gamma_1.4",
                                     "temperature_3000K", "temperature_9000K",
                                     "shading_0.5_at_90deg"])
    strip("strip_occlusion.png", ["original", "eyes_5pct", "eyes_10pct", "eyes_20pct",
                                  "mouth_10pct", "mask_band"])
    strip("strip_resolution.png", ["original", "downscale_160px", "downscale_112px",
                                   "downscale_80px", "downscale_40px"])

    copy_references()
    for ref in INV["references"]:
        name = ref["identity"].replace("person_", "")
        crop = cv2.imread(str(PROJECT / ref["aligned_crop"]))
        if crop is not None:
            cv2.imwrite(str(IMG / f"aligned_{name}.png"), crop)

    (OUT / "index.html").write_text(build_html())
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    n = sum(1 for f in OUT.rglob("*") if f.is_file())
    print(f"wrote {OUT/'index.html'}  ({n} files, {total/1024:.0f} KiB)")


if __name__ == "__main__":
    main()
