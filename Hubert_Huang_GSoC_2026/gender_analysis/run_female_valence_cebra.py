#!/usr/bin/env python
"""
scripts/run_female_valence_cebra.py
===================================
Female-only positive vs negative "valence" analysis.

METADATA SOURCE (explicit, not inferred):
  file:  Filtered Data/Demographic Information/Demographic Information/
         Updated-Dyad-Pair-DemographicInfo-asof-072326.xlsx, sheet 'Full-Info'
  columns: 'InteractionQuality' (values: 'Satisfactory' / 'Deficient')
           'IntRating_code'      (1 = Satisfactory, 0 = Deficient)
  This is a rated interaction-quality label, not a column literally named
  "valence". We map Satisfactory -> positive (1), Deficient -> negative (0)
  and document that mapping explicitly here and in every report of these
  results; this is a disclosed interpretation of the closest available
  explicit label, not an inference from filenames/order/content.
  The label is per (dyad, speaker-direction) row -- i.e. it rates that
  person's SPEAKING turn in that dyad -- so it is attached to that
  participant's own `..._speak.edf` recording, restricted to FEMALE speakers
  (Speaker Gender == F) with a non-null IntRating_code.

WINDOWING:
  From each qualifying speak recording (uniform 250 Hz, verified to be at
  least 297 s here, i.e. >> 240 s): first 2 minutes (30000 samples) and last
  2 minutes (30000 samples). Overlap is explicitly checked and would be
  handled by taking a single non-duplicated window centered in the
  recording, but no file in this dataset triggers that path (all are
  >= 297 s > 240 s), which is logged.

CEBRA config matches the rest of this project exactly (offset10-model, 3D,
cosine, time_delta, batch 512, lr 3e-4, temp 1.12, offsets 10, 5000
iterations, seed 0, per-channel z-score, 64 EEG channels).

Because valence (like AQ magnitude, gender-type, and gender) is constant
within a participant across both of their extracted windows, a full
retraining permutation control is run: valence labels are shuffled across
whole PARTICIPANTS (both windows keep the same permuted label), class counts
preserved, CEBRA retrained from scratch per permutation (>=5 retrainings).
"""
from __future__ import annotations
import argparse, csv, json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import torch, cebra
    from cebra import CEBRA
    from cebra.integrations.sklearn import metrics as cmetrics
except ImportError:
    import sys
    sys.exit("need torch + cebra")
import mne
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, balanced_accuracy_score, f1_score,
                             recall_score, confusion_matrix)

mne.set_log_level("ERROR")
CEBRA_KW = dict(model_architecture="offset10-model", batch_size=512, learning_rate=3e-4,
                temperature=1.12, conditional="time_delta", output_dimension=3,
                distance="cosine", device="cuda_if_available", verbose=True, time_offsets=10)
FAULTY = {(38, 98), (41, 104)}
WIN_SEC = 120.0
SUB = 60000
LABELS = ["negative", "positive"]


def eeg_picks(names):
    return [i for i, c in enumerate(names)
            if not any(t in c.upper() for t in ("VREF", "STATUS", "TRIGGER", "STI"))]


def build_label_table(xlsx_path, gender_json):
    df = pd.read_excel(xlsx_path, sheet_name="Full-Info")
    df = df[df["pair_id"].notna()].copy()

    def parse_pair(pid):
        parts = re.split("_", str(pid))
        return frozenset(parts) if len(parts) == 2 else None

    pairmembers_to_dyad = {}
    for _, r in df.iterrows():
        if pd.notna(r["dyad_id"]):
            pm = parse_pair(r["pair_id"])
            if pm:
                pairmembers_to_dyad[pm] = r["dyad_id"]
    df["pair_members"] = df["pair_id"].apply(parse_pair)
    df["dyad_id_filled"] = df.apply(
        lambda r: r["dyad_id"] if pd.notna(r["dyad_id"])
        else pairmembers_to_dyad.get(r["pair_members"]), axis=1)
    df["dyad_num"] = df["dyad_id_filled"].astype(str).str.extract(r"dyad0*(\d+)").astype(float)
    df["sid_int"] = pd.to_numeric(df["Speaker ID"], errors="coerce")

    pg = {int(k): v for k, v in json.load(open(gender_json))["participant_gender"].items()}

    rows = []
    for _, r in df.iterrows():
        if pd.isna(r["sid_int"]) or pd.isna(r["dyad_num"]) or pd.isna(r["IntRating_code"]):
            continue
        sid = int(r["sid_int"])
        dnum = int(r["dyad_num"])
        if pg.get(sid) != "F" or (dnum, sid) in FAULTY:
            continue
        rows.append({"dyad": dnum, "pid": sid, "label": int(r["IntRating_code"]),
                     "quality": r["InteractionQuality"], "pair_id": r["pair_id"]})
    return rows


def grouped_knn(E, y, g, k=5):
    yt, yp = [], []
    for q in sorted(set(g.tolist())):
        te = g == q
        tr = ~te
        if te.sum() == 0 or len(set(y[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(E[tr])
        m = KNeighborsClassifier(k).fit(sc.transform(E[tr]), y[tr])
        yp.append(m.predict(sc.transform(E[te])))
        yt.append(y[te])
    if not yt:
        return None, None
    return np.concatenate(yt), np.concatenate(yp)


def evaluate(emb, y, g, seed):
    rng = np.random.RandomState(seed)
    sub = rng.choice(len(emb), min(SUB, len(emb)), replace=False)
    sub.sort()
    E, Y, G = emb[sub], y[sub], g[sub]
    yt, yp = grouped_knn(E, Y, G)
    if yt is None:
        return {"note": "no valid folds"}, sub
    out = {"accuracy": float((yt == yp).mean()),
           "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
           "macro_f1": float(f1_score(yt, yp, average="macro")),
           "per_class_recall": {LABELS[i]: float(v) for i, v in enumerate(
               recall_score(yt, yp, average=None, labels=[0, 1], zero_division=0))},
           "confusion_matrix": confusion_matrix(yt, yp, labels=[0, 1]).tolist(),
           "majority_chance": float(np.bincount(Y, minlength=2).max() / len(Y)),
           "n_groups": int(len(set(G.tolist())))}
    try:
        s2 = rng.choice(len(E), min(10000, len(E)), replace=False)
        out["silhouette"] = float(silhouette_score(E[s2], Y[s2]))
    except Exception:
        out["silhouette"] = None
    return out, sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--valence-xlsx", required=True)
    ap.add_argument("--gender-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-iterations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perm-seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "permutation_mappings").mkdir(exist_ok=True)
    srcdir = Path(args.input_dir)

    entries = build_label_table(args.valence_xlsx, args.gender_json)
    win = int(WIN_SEC * 250)

    windows, meta, overlap_log = [], [], []
    cur = 0
    blocks = []
    for e in entries:
        fp = srcdir / f"fullband250_dyad{e['dyad']}_{e['pid']}_speak.edf"
        if not fp.exists():
            print(f"MISSING FILE, skipping: {fp.name}", flush=True)
            continue
        raw = mne.io.read_raw_edf(str(fp), preload=True, verbose="ERROR")
        picks = eeg_picks(raw.ch_names)
        data = raw.get_data()[picks]
        n = data.shape[1]
        raw.close()
        overlap = (n - win) < win  # start window end vs end window start
        if overlap:
            overlap_log.append({"file": fp.name, "n_samples": n,
                                "note": "overlap detected -- using single centered window, "
                                        "not duplicated start+end"})
            s = max(0, (n - win) // 2)
            segs = [("center", data[:, s:s + win])]
        else:
            segs = [("start", data[:, :win]), ("end", data[:, n - win:n])]
        for tag, seg in segs:
            X = seg.T.astype(np.float32)
            blocks.append(X)
            meta.append({"file": fp.name, "dyad": e["dyad"], "pid": e["pid"],
                         "window": tag, "label": e["label"], "quality": e["quality"],
                         "start": cur, "end": cur + X.shape[0]})
            cur += X.shape[0]

    X = np.concatenate(blocks)
    del blocks
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True)
    sd[sd == 0] = 1
    X -= mu
    X /= sd
    X = np.ascontiguousarray(X, np.float32)

    y = np.zeros(len(X), np.int64)
    pid_v = np.zeros(len(X), np.int64)
    dy_v = np.zeros(len(X), np.int64)
    for m in meta:
        s, e = m["start"], m["end"]
        y[s:e] = m["label"]
        pid_v[s:e] = m["pid"]
        dy_v[s:e] = m["dyad"]
    n_ppl = len({m["pid"] for m in meta})
    n_dyads = len({m["dyad"] for m in meta})
    n_files = len({m["file"] for m in meta})
    n_pos_files = len({(m["pid"], m["file"]) for m in meta if m["label"] == 1})
    n_neg_files = n_files - n_pos_files
    class_counts = np.bincount(y, minlength=2).tolist()
    print(f"files={n_files} (pos={n_pos_files}, neg={n_neg_files})  windows={len(meta)}  "
          f"participants={n_ppl}  dyads={n_dyads}  samples={len(X)}  "
          f"class_counts(neg,pos)={class_counts}  overlaps={len(overlap_log)}", flush=True)

    with open(out / "sample_metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "dyad", "pid", "window", "label",
                                          "quality", "start", "end"])
        w.writeheader()
        w.writerows(meta)
    prov = {"metadata_source": str(args.valence_xlsx) + " [Full-Info] "
                              "InteractionQuality/IntRating_code",
            "label_mapping": {"Satisfactory": 1, "Deficient": 0,
                              "positive": 1, "negative": 0},
            "window_seconds": WIN_SEC, "window_samples": win,
            "n_files": n_files, "n_positive_files": n_pos_files,
            "n_negative_files": n_neg_files, "n_windows": len(meta),
            "n_participants": n_ppl, "n_dyads": n_dyads, "n_samples": int(len(X)),
            "class_counts_neg_pos": class_counts,
            "majority_chance": float(max(class_counts) / sum(class_counts)),
            "overlap_cases": overlap_log,
            "cebra_config": dict(CEBRA_KW, max_iterations=args.max_iterations,
                                 seed=args.seed)}
    json.dump(prov, open(out / "dataset_provenance.json", "w"), indent=2)

    records = []
    rd = out / "real"
    if not (rd / "metrics.json").exists():
        rd.mkdir(exist_ok=True)
        t0 = time.time()
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = CEBRA(max_iterations=args.max_iterations, **CEBRA_KW)
        model.fit(X, y)
        model.save(str(rd / "model.pt"))
        emb = model.transform(X).astype(np.float32)
        np.save(rd / "embedding.npy", emb)
        lh = list(getattr(model, "state_dict_", {}).get("loss", []))
        fl = float(lh[-1]) if lh else None
        try:
            gof = float(cmetrics.goodness_of_fit_score(model, X, y))
        except Exception:
            gof = None
        m, sub = evaluate(emb, y, pid_v, args.seed)
        m2, _ = evaluate(emb, y, dy_v, args.seed)
        res = {"run": "real", "final_loss": fl, "goodness_of_fit_bits": gof,
               "LOPO_by_participant": m, "LODO_by_dyad": m2,
               "train_seconds": round(time.time() - t0, 1)}
        json.dump(res, open(rd / "metrics.json", "w"), indent=2)
        np.save(rd / "eval_indices.npy", sub)
        print(f"REAL loss={fl:.4f} GoF={gof:.4f} LOPO_acc={m.get('accuracy')} "
              f"bal={m.get('balanced_accuracy')} sil={m.get('silhouette')} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    records.append(json.load(open(rd / "metrics.json")))

    ppl = sorted({m["pid"] for m in meta})
    lab_by_person = {}
    for m in meta:
        lab_by_person[m["pid"]] = m["label"]
    labels_arr = np.array([lab_by_person[p] for p in ppl])
    for ps in args.perm_seeds:
        pdir = out / f"perm{ps}"
        if (pdir / "metrics.json").exists():
            print(f"SKIP perm{ps}", flush=True)
            records.append(json.load(open(pdir / "metrics.json")))
            continue
        pdir.mkdir(parents=True, exist_ok=True)
        perm = np.random.RandomState(9000 + ps).permutation(len(ppl))
        pmap = {q: int(labels_arr[perm[i]]) for i, q in enumerate(ppl)}
        assert sorted(pmap.values()) == sorted(labels_arr.tolist())
        yq = np.empty_like(y)
        for m in meta:
            yq[m["start"]:m["end"]] = pmap[m["pid"]]
        assert np.bincount(yq, minlength=2).tolist() == class_counts
        n_relab = sum(1 for p in pmap if pmap[p] != lab_by_person[p])
        json.dump({"perm_seed": ps, "n_relabelled": n_relab,
                   "permuted": {str(p): pmap[p] for p in ppl},
                   "real": {str(p): lab_by_person[p] for p in ppl}},
                  open(out / "permutation_mappings" / f"perm{ps}.json", "w"), indent=2)
        print(f"\n=== VALENCE PERMUTATION {ps} ({n_relab}/{len(ppl)} relabelled) ===",
              flush=True)
        t0 = time.time()
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = CEBRA(max_iterations=args.max_iterations, **CEBRA_KW)
        model.fit(X, yq)
        model.save(str(pdir / "model.pt"))
        emb = model.transform(X).astype(np.float32)
        lh = list(getattr(model, "state_dict_", {}).get("loss", []))
        fl = float(lh[-1]) if lh else None
        try:
            gof = float(cmetrics.goodness_of_fit_score(model, X, yq))
        except Exception:
            gof = None
        m, _ = evaluate(emb, yq, pid_v, args.seed)
        res = {"run": f"perm{ps}", "perm_seed": ps, "n_relabelled": n_relab,
               "final_loss": fl, "goodness_of_fit_bits": gof,
               "LOPO_by_participant": m, "train_seconds": round(time.time() - t0, 1)}
        json.dump(res, open(pdir / "metrics.json", "w"), indent=2)
        records.append(res)
        print(f"perm{ps}: loss={fl:.4f} GoF={gof:.4f} acc={m.get('accuracy')} "
              f"bal={m.get('balanced_accuracy')} sil={m.get('silhouette')} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
        del emb, model

    real_acc = records[0]["LOPO_by_participant"]["accuracy"]
    real_gof = records[0]["goodness_of_fit_bits"]
    real_sil = records[0]["LOPO_by_participant"]["silhouette"]
    perms_only = [r for r in records if r["run"] != "real"]
    accs = np.array([r["LOPO_by_participant"]["accuracy"] for r in perms_only])
    gofs = np.array([r["goodness_of_fit_bits"] for r in perms_only])
    sils = np.array([r["LOPO_by_participant"]["silhouette"] for r in perms_only])
    summary = {}
    for name, real_v, null_v, higher_better in [
            ("LOPO_accuracy", real_acc, accs, True),
            ("goodness_of_fit_bits", real_gof, gofs, True),
            ("silhouette", real_sil, sils, True)]:
        if higher_better:
            beat = int((null_v >= real_v).sum())
        else:
            beat = int((null_v <= real_v).sum())
        p = (1 + beat) / (1 + len(null_v))
        summary[name] = {"real": real_v, "null_mean": float(null_v.mean()),
                         "null_std": float(null_v.std()), "null_min": float(null_v.min()),
                         "null_max": float(null_v.max()), "n_perms": len(null_v),
                         "n_beating_real": beat, "empirical_p": p}
    json.dump(summary, open(out / "real_vs_null_summary.json", "w"), indent=2)
    print("\n=== REAL vs NULL ===")
    for k, v in summary.items():
        print(f"  {k}: real={v['real']:.4f} null={v['null_mean']:.4f}+-{v['null_std']:.4f} "
              f"p={v['empirical_p']:.4f} ({v['n_beating_real']}/{v['n_perms']} beat real)")
    print("VALENCE_DONE")


if __name__ == "__main__":
    main()
