#!/usr/bin/env python
"""
examples/build_lightning_talk_pptx.py
=======================================
Builds the 3-minute ML4SCI GSoC 2026 lightning-talk deck (NeuroDyads,
Hubert Huang) as an editable PPTX with embedded speaker notes.

All numeric claims are read from already-computed, finalized result files
on disk (no model is retrained or rerun here). All figures are copies of
existing project PNGs; nothing is regenerated from raw data.

This is a one-off "build my own final deck" example, not a general CLI tool:
paths are all relative to this repo (ROOT = the folder above examples/, i.e.
Hubert_Huang_GSoC_2026/), and it reads/validates its inputs at import time
(so it exits immediately with a clear "MISSING REQUIRED FIGURE"/file-not-found
error unless run against a `results/` tree that already contains the full
finalized 250 Hz outputs described below). Requires python-pptx
(`pip install python-pptx`), which is not needed by any other script in this
submission.

Sources verified immediately before building:
  - results/aug4_pipeline/rerun250/dyad_permutation/real_vs_null_summary.csv
  - results/aug4_pipeline/rerun250/fullband_baseline/metrics.json
  - results/aug4_pipeline/individual_aq_role/followup_checks.json
  - results/gender_analysis/participant_states/metrics.json
"""
from __future__ import annotations

import json
import csv
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "presentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PPTX = OUT_DIR / "ML4Sci_GSoC2026_NeuroDyads_LightningTalk.pptx"

FIG_AQ_DYAD_NULL = ROOT / "results/aug4_pipeline/overleaf_figures_250hz/fullband_speakerfirst_permutation_5nn_null.png"
FIG_AQ_PARTICIPANT_NULL = ROOT / "results/aug4_pipeline/individual_aq_role/individual_aq_permutation_null.png"
FIG_MANIFOLD = ROOT / "results/aug4_pipeline/rerun250/aq_delta6/aq_delta6_embedding.png"

for f in (FIG_AQ_DYAD_NULL, FIG_AQ_PARTICIPANT_NULL, FIG_MANIFOLD):
    if not f.exists():
        raise SystemExit(f"MISSING REQUIRED FIGURE: {f}")

# ---------------------------------------------------------------- verify numbers
with open(ROOT / "results/aug4_pipeline/rerun250/dyad_permutation/real_vs_null_summary.csv") as fh:
    row = next(r for r in csv.DictReader(fh) if r["metric"] == "knn5_leave_one_dyad_out")
AQ_DYAD_REAL = float(row["real"])
AQ_DYAD_NULL_MEAN = float(row["null_mean"])
AQ_DYAD_NULL_STD = float(row["null_std"])
AQ_DYAD_P = float(row["empirical_p"])

baseline = json.load(open(ROOT / "results/aug4_pipeline/rerun250/fullband_baseline/metrics.json"))
AQ_DYAD_CHANCE = float(baseline["chance"])

followup = json.load(open(ROOT / "results/aug4_pipeline/individual_aq_role/followup_checks.json"))
part_perm = followup["dyad_permutation"]
AQ_PART_REAL = float(part_perm["real"])
AQ_PART_NULL_MEAN = float(part_perm["null_mean"])
AQ_PART_NULL_STD = float(part_perm["null_std"])
AQ_PART_P = float(part_perm["empirical_p"])

state = json.load(open(ROOT / "results/gender_analysis/participant_states/metrics.json"))
STATE_ACC = state["state_decoding_dyad_holdout"]["accuracy"]
STATE_CHANCE = state["state_decoding_dyad_holdout"]["majority_chance"]

print("Verified numbers used on slides:")
print(f"  AQ (dyad)   real={AQ_DYAD_REAL:.4f}  chance={AQ_DYAD_CHANCE:.4f}  "
      f"null={AQ_DYAD_NULL_MEAN:.4f}+/-{AQ_DYAD_NULL_STD:.4f}  p={AQ_DYAD_P:.4f}")
print(f"  AQ (indiv)  real={AQ_PART_REAL:.4f}  null={AQ_PART_NULL_MEAN:.4f}+/-{AQ_PART_NULL_STD:.4f}  "
      f"p={AQ_PART_P:.4f}")
print(f"  State decoding: acc={STATE_ACC:.4f}  chance={STATE_CHANCE:.4f}")

# sanity-check against the numbers specified in the task (rounded to what's on the slide)
assert round(AQ_DYAD_REAL, 3) == 0.759, AQ_DYAD_REAL
assert round(AQ_DYAD_CHANCE, 3) == 0.545, AQ_DYAD_CHANCE
assert round(AQ_DYAD_NULL_MEAN, 3) == 0.756, AQ_DYAD_NULL_MEAN
assert round(AQ_DYAD_NULL_STD, 3) == 0.009, AQ_DYAD_NULL_STD
assert round(AQ_DYAD_P, 2) == 0.50, AQ_DYAD_P
assert round(AQ_PART_REAL, 3) == 0.247, AQ_PART_REAL
assert round(AQ_PART_NULL_MEAN, 3) == 0.116, AQ_PART_NULL_MEAN
assert round(AQ_PART_NULL_STD, 3) == 0.020, AQ_PART_NULL_STD
assert round(AQ_PART_P, 4) == 0.0099, AQ_PART_P
print("All slide numbers match finalized results exactly. Proceeding.\n")

# ---------------------------------------------------------------- style constants
FONT = "Calibri"
NAVY = RGBColor(0x14, 0x39, 0x5B)
DARK = RGBColor(0x21, 0x25, 0x29)
GRAY = RGBColor(0x6C, 0x75, 0x7D)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED_BG = RGBColor(0xFD, 0xEC, 0xEA)
RED_TXT = RGBColor(0x8B, 0x2E, 0x2A)
GREEN_BG = RGBColor(0xE7, 0xF6, 0xEC)
GREEN_TXT = RGBColor(0x1E, 0x56, 0x31)
ACCENT_ORANGE = RGBColor(0xC0, 0x39, 0x2B)
STEP_BOX_FILL = RGBColor(0x1B, 0x4F, 0x72)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
FOOTER_TEXT = "Google Summer of Code 2026  |  ML4SCI NeuroDyads  |  Hubert Huang"

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK = prs.slide_layouts[6]


def add_slide():
    return prs.slides.add_slide(BLANK)


def set_run(run, text, size, bold=False, color=DARK, italic=False, font=FONT):
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = font
    run.font.color.rgb = color


def add_textbox(slide, left, top, width, height, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    return tb, tf


def add_title(slide, text, size=40, color=NAVY, top=Inches(0.35), height=Inches(1.0)):
    tb, tf = add_textbox(slide, Inches(0.5), top, Inches(12.333), height)
    p = tf.paragraphs[0]
    r = p.add_run()
    set_run(r, text, size, bold=True, color=color)
    return tb


def add_bullets(slide, items, left, top, width, height, size=24, color=DARK,
                 space_after=14, marker="•  "):
    tb, tf = add_textbox(slide, left, top, width, height)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        r = p.add_run()
        set_run(r, marker + item, size, color=color)
    return tb


def add_footer(slide):
    tb, tf = add_textbox(slide, Inches(0.5), Inches(7.08), Inches(12.333), Inches(0.35))
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    set_run(r, FOOTER_TEXT, 12, color=GRAY, italic=True)
    line = slide.shapes.add_connector(1, Inches(0.5), Inches(7.02), Inches(12.833), Inches(7.02))
    line.line.color.rgb = RGBColor(0xDD, 0xDD, 0xDD)
    line.line.width = Pt(0.75)


def add_picture_fit(slide, path, left, top, max_w, max_h):
    """Place an image, scaled to fit within (max_w, max_h), preserving aspect ratio,
    centered within that box."""
    from PIL import Image
    with Image.open(path) as im:
        iw, ih = im.size
    aspect = iw / ih
    box_aspect = max_w / max_h
    if aspect > box_aspect:
        w = max_w
        h = int(max_w / aspect)
    else:
        h = max_h
        w = int(max_h * aspect)
    x = left + (max_w - w) // 2
    y = top + (max_h - h) // 2
    return slide.shapes.add_picture(str(path), x, y, width=w, height=h)


def set_notes(slide, text):
    notes = slide.notes_slide
    notes.notes_text_frame.text = text


# ============================================================== SLIDE 1
s1 = add_slide()
add_title(s1, "What does a neural embedding actually learn?", size=34)

bullets_1 = [
    "EEG hyperscanning: two people recorded together during real conversation",
    "CEBRA compresses 64-channel EEG into a 3D neural manifold",
    "My question: does it encode real traits — or just who's talking?",
]
add_bullets(s1, bullets_1, Inches(0.5), Inches(1.7), Inches(6.0), Inches(4.5), size=24)

add_picture_fit(s1, FIG_MANIFOLD, Inches(6.8), Inches(1.7), Inches(6.0), Inches(4.6))
tb, tf = add_textbox(s1, Inches(6.8), Inches(6.35), Inches(6.0), Inches(0.5))
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
r = p.add_run()
set_run(r, "A real CEBRA embedding of EEG hyperscanning data", 14, color=GRAY, italic=True)

add_footer(s1)
set_notes(s1,
    "We record two people's brain activity — EEG — at the same time during a "
    "real conversation. That's called hyperscanning. We use a tool called CEBRA to "
    "compress each person's 64-channel EEG signal into a compact, three-dimensional "
    "neural manifold, shown on the right. My core question this summer, for my ML4Sci "
    "GSoC project: does that manifold capture something scientifically meaningful — "
    "like a personality trait — or does it just learn who's who?")

# ============================================================== SLIDE 2
s2 = add_slide()
add_title(s2, "What I built", size=34)

steps = ["EEG", "250 Hz\nstandardize", "Speaker/\nListener\nalign", "CEBRA",
         "Grouped\ndecoding", "Geometry", "Permutation\ncontrols"]
n = len(steps)
box_w = Inches(1.5)
gap = Inches(0.30)
total_w = box_w * n + gap * (n - 1)
start_x = int((SLIDE_W - total_w) / 2)
box_y = Inches(1.75)
box_h = Inches(1.15)

for i, label in enumerate(steps):
    x = start_x + i * (box_w + gap)
    shp = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, box_y, box_w, box_h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = STEP_BOX_FILL
    shp.line.color.rgb = WHITE
    shp.line.width = Pt(1.5)
    tf = shp.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(2)
    tf.margin_right = Pt(2)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    lines = label.split("\n")
    for li, ln in enumerate(lines):
        p = tf.paragraphs[0] if li == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        set_run(r, ln, 13, bold=True, color=WHITE)
    if i < n - 1:
        ax = x + box_w + Emu(0)
        arrow_tb, arrow_tf = add_textbox(s2, x + box_w, box_y, gap, box_h, anchor=MSO_ANCHOR.MIDDLE)
        ap = arrow_tf.paragraphs[0]
        ap.alignment = PP_ALIGN.CENTER
        ar = ap.add_run()
        set_run(ar, "→", 20, bold=True, color=NAVY)

bullets_2 = [
    "33-dyad speaker-first pipeline, resampled to a uniform 250 Hz",
    "Leave-one-dyad-out evaluation — never train and test on the same pair",
    "Frequency-band ablations, aperiodic analysis, and full CEBRA retraining under label permutations",
]
add_bullets(s2, bullets_2, Inches(0.5), Inches(3.4), Inches(12.333), Inches(3.0), size=24)

add_footer(s2)
set_notes(s2,
    "Here's the pipeline I built. Every recording is standardized to 250 Hz, speaker and "
    "listener channels are aligned, and then CEBRA learns the embedding. I evaluate "
    "everything with leave-one-dyad-out cross-validation, so the model is never tested "
    "on a pair it trained on. I also ran frequency-band ablations — removing delta, "
    "theta, alpha, beta, and gamma — plus a non-oscillatory, aperiodic-only analysis. "
    "Critically, I don't just reshuffle labels for the decoder. I fully retrain CEBRA "
    "from scratch under permuted labels, five or more times, to build a genuine null "
    "distribution.")

# ============================================================== SLIDE 3
s3 = add_slide()
add_title(s3, "High accuracy was not enough.", size=36, color=ACCENT_ORANGE, height=Inches(0.85))

col_w = Inches(5.9)
col_gap = Inches(0.53)
col1_x = Inches(0.5)
col2_x = col1_x + col_w + col_gap
label_y = Inches(1.35)
img_y = Inches(1.85)
img_h = Inches(3.05)
stat_y = Inches(5.0)
stat_h = Inches(1.65)

tb, tf = add_textbox(s3, col1_x, label_y, col_w, Inches(0.4))
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
set_run(p.add_run(), "AQ Magnitude — Dyad Level", 20, bold=True, color=DARK)

tb, tf = add_textbox(s3, col2_x, label_y, col_w, Inches(0.4))
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
set_run(p.add_run(), "AQ — Participant Level", 20, bold=True, color=DARK)

add_picture_fit(s3, FIG_AQ_DYAD_NULL, col1_x, img_y, col_w, img_h)
add_picture_fit(s3, FIG_AQ_PARTICIPANT_NULL, col2_x, img_y, col_w, img_h)

box1 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, col1_x, stat_y, col_w, stat_h)
box1.fill.solid(); box1.fill.fore_color.rgb = RED_BG
box1.line.color.rgb = RED_TXT; box1.line.width = Pt(1.5)
tf1 = box1.text_frame; tf1.word_wrap = True; tf1.vertical_anchor = MSO_ANCHOR.MIDDLE
tf1.margin_left = Pt(10); tf1.margin_right = Pt(10)
lines1 = [
    (f"LODO: {AQ_DYAD_REAL:.3f}   (chance {AQ_DYAD_CHANCE:.3f})", 17, False),
    (f"Retrained null: {AQ_DYAD_NULL_MEAN:.3f} ± {AQ_DYAD_NULL_STD:.3f}", 17, False),
    (f"p = {AQ_DYAD_P:.2f}  —  FAILS CONTROL", 19, True),
]
for i, (txt, sz, bold) in enumerate(lines1):
    p = tf1.paragraphs[0] if i == 0 else tf1.add_paragraph()
    p.alignment = PP_ALIGN.CENTER
    set_run(p.add_run(), txt, sz, bold=bold, color=RED_TXT)

box2 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, col2_x, stat_y, col_w, stat_h)
box2.fill.solid(); box2.fill.fore_color.rgb = GREEN_BG
box2.line.color.rgb = GREEN_TXT; box2.line.width = Pt(1.5)
tf2 = box2.text_frame; tf2.word_wrap = True; tf2.vertical_anchor = MSO_ANCHOR.MIDDLE
tf2.margin_left = Pt(10); tf2.margin_right = Pt(10)
lines2 = [
    (f"Real: {AQ_PART_REAL:.3f}   Null: {AQ_PART_NULL_MEAN:.3f} ± {AQ_PART_NULL_STD:.3f}", 17, False),
    (f"p = {AQ_PART_P:.4f}", 17, False),
    ("SURVIVES CONTROL", 19, True),
]
for i, (txt, sz, bold) in enumerate(lines2):
    p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
    p.alignment = PP_ALIGN.CENTER
    set_run(p.add_run(), txt, sz, bold=bold, color=GREEN_TXT)

add_footer(s3)
set_notes(s3,
    "Here's the main finding. Using AQ — the Autism Spectrum Quotient — as a "
    "dyad-level label, CEBRA decodes it at 75.9% accuracy, above the 54.5% baseline. "
    "Looks great. But when I fully retrain CEBRA five times on randomly permuted dyad "
    "labels, the null comes out at 75.6% — nearly identical, p equals 0.50. The model "
    "isn't learning AQ; it's learning dyad identity. This is why decoder-only permutation "
    "tests aren't enough — you must retrain the whole encoder. But it's not all "
    "negative: at the participant level, real accuracy is 24.7% against a null of 11.6% "
    "— p equals 0.0099. That result is real.")

# ============================================================== SLIDE 4
s4 = add_slide()
add_title(s4, "Takeaways", size=36)

bullets_4 = [
    "Contrastive embeddings can strongly encode identity — not just your intended trait",
    "Group-aware splits aren't enough when the encoder saw the labels; retraining-based permutation controls matter",
    "Individual- and state-level analyses offer a cleaner path forward",
]
add_bullets(s4, bullets_4, Inches(0.5), Inches(1.55), Inches(12.333), Inches(3.0), size=25, space_after=20)

tb, tf = add_textbox(s4, Inches(0.5), Inches(4.85), Inches(12.333), Inches(1.7))
sub_lines = [
    f"State decoding reliably distinguishes rest from interaction "
    f"({STATE_ACC*100:.0f}% vs. {STATE_CHANCE*100:.0f}% chance)",
    "Contributed to a NeurIPS workshop submission",
    "Code: ML4SCI/NeuroDyads",
]
for i, ln in enumerate(sub_lines):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.space_after = Pt(10)
    set_run(p.add_run(), ln, 17, color=GRAY)

add_footer(s4)
set_notes(s4,
    "Three takeaways. First, contrastive embeddings can strongly encode identity, not "
    "just your intended trait. Second, group-aware decoder splits alone aren't enough "
    "when the encoder itself saw the labels during training — full retraining "
    "permutation controls are essential. Third, individual- and state-level analyses "
    "look more promising: state decoding reliably separates resting from active "
    "interaction, well above chance. This work contributed to a NeurIPS workshop "
    "submission, and all code is available at ML4Sci slash NeuroDyads. Thanks to my "
    "mentors and to ML4Sci for a great summer.")

# ---------------------------------------------------------------- save
prs.save(str(OUT_PPTX))
print(f"\nSaved: {OUT_PPTX}")

# ---------------------------------------------------------------- word count / timing check
notes_texts = [s.notes_slide.notes_text_frame.text for s in prs.slides]
total_words = sum(len(t.split()) for t in notes_texts)
print("\nSpeaker notes word counts:")
for i, t in enumerate(notes_texts, 1):
    wc = len(t.split())
    print(f"  Slide {i}: {wc} words")
print(f"  TOTAL: {total_words} words")
for wpm in (130, 140, 150):
    secs = total_words / wpm * 60
    print(f"  @ {wpm} wpm -> {secs:.0f}s ({secs/60:.2f} min)")
