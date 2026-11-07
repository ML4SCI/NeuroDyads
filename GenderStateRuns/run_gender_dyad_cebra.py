#!/usr/bin/env python
"""
scripts/run_gender_dyad_cebra.py
================================
Dyad-level gender-composition analysis (MM / FF / MF) on the finalized 250 Hz
speaker-first dataset, using the project standard CEBRA configuration.

Gender type is CONSTANT within a dyad, exactly as AQ magnitude was, so the same
dyad-level permutation control applies: labels are permuted across whole dyads,
class counts preserved, and CEBRA retrained from scratch each time. Nothing is
claimed unless the real run separates from that retraining null.

Both role mappings of a dyad carry the same gender-type label, and MF is MF
regardless of who is speaking.
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np
try:
    import torch, cebra
    from cebra import CEBRA
    from cebra.integrations.sklearn import metrics as cmetrics
except ImportError:
    import sys; sys.exit("need torch + cebra")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, confusion_matrix,
                             balanced_accuracy_score, f1_score, recall_score)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CEBRA_KW = dict(model_architecture="offset10-model", batch_size=512, learning_rate=3e-4,
                temperature=1.12, conditional="time_delta", output_dimension=3,
                distance="cosine", device="cuda_if_available", verbose=True, time_offsets=10)
SUB = 60000
TYPES = ["MM", "FF", "MF"]


def to_tc(x):
    return x.T if x.shape[0] < x.shape[1] else x


def wrap180(x):
    return (x + 180) % 360 - 180


def lodo(E, y, g, k=5):
    yt, yp, folds = [], [], []
    for d in sorted(set(g.tolist())):
        te = g == d; tr = ~te
        if te.sum() == 0 or len(set(y[tr].tolist())) < 2:
            folds.append({"dyad": int(d), "skipped": True}); continue
        sc = StandardScaler().fit(E[tr])
        m = KNeighborsClassifier(k).fit(sc.transform(E[tr]), y[tr])
        p = m.predict(sc.transform(E[te]))
        yt.append(y[te]); yp.append(p)
        folds.append({"dyad": int(d), "n": int(te.sum()), "true": int(y[te][0]),
                      "acc": float((p == y[te]).mean())})
    if not yt:
        return None, None, folds
    return np.concatenate(yt), np.concatenate(yp), folds


def evaluate(emb, y, g, seed, nclass=3):
    rng = np.random.RandomState(seed)
    sub = rng.choice(len(emb), min(SUB, len(emb)), replace=False); sub.sort()
    E, Y, G = emb[sub], y[sub], g[sub]
    yt, yp, folds = lodo(E, Y, G)
    out = {"lodo_acc": float((yt == yp).mean()),
           "lodo_balanced": float(balanced_accuracy_score(yt, yp)),
           "lodo_macro_f1": float(f1_score(yt, yp, average="macro")),
           "per_class_recall": {TYPES[i]: float(v) for i, v in enumerate(
               recall_score(yt, yp, average=None, labels=list(range(nclass)),
                            zero_division=0))},
           "confusion_matrix": confusion_matrix(yt, yp,
                                                labels=list(range(nclass))).tolist(),
           "majority_chance": float(np.bincount(Y, minlength=nclass).max() / len(Y)),
           "class_counts_eval": np.bincount(Y, minlength=nclass).tolist()}
    idx = rng.permutation(len(E)); cut = int(0.7 * len(E))
    sc = StandardScaler().fit(E[idx[:cut]])
    out["random_split_acc"] = float(KNeighborsClassifier(5).fit(
        sc.transform(E[idx[:cut]]), Y[idx[:cut]]).score(
        sc.transform(E[idx[cut:]]), Y[idx[cut:]]))
    try:
        s2 = rng.choice(len(E), min(10000, len(E)), replace=False)
        out["silhouette"] = (float(silhouette_score(E[s2], Y[s2]))
                             if len(set(Y[s2].tolist())) > 1 else None)
    except Exception:
        out["silhouette"] = None
    Ec = E - np.median(E, axis=0)
    x_, y_, z_ = Ec[:, 0], Ec[:, 1], Ec[:, 2]
    r = np.sqrt(x_ ** 2 + y_ ** 2 + z_ ** 2)
    lon = np.degrees(np.arctan2(y_, x_))
    lat = np.degrees(np.arctan2(z_, np.sqrt(x_ ** 2 + y_ ** 2)))
    cm = np.degrees(np.arctan2(np.sin(np.radians(lon)).mean(),
                               np.cos(np.radians(lon)).mean()))
    return out, sub, folds, (wrap180(lon - cm), lat, r, float(cm))


def pairwise(emb, y, g, seed, sub):
    E, Y, G = emb[sub], y[sub], g[sub]
    rows = []
    for a in range(3):
        for b in range(a + 1, 3):
            m = (Y == a) | (Y == b)
            nd = len(set(G[m].tolist()))
            if m.sum() < 20 or nd < 3:
                rows.append({"pair": TYPES[a] + " vs " + TYPES[b],
                             "n_samples": int(m.sum()), "n_dyads": int(nd),
                             "note": "too few dyads for leave-one-dyad-out"})
                continue
            yb = (Y[m] == b).astype(np.int64)
            yt, yp, _ = lodo(E[m], yb, G[m])
            if yt is None:
                rows.append({"pair": TYPES[a] + " vs " + TYPES[b],
                             "n_samples": int(m.sum()), "n_dyads": int(nd),
                             "note": "no valid folds"})
                continue
            acc = float((yt == yp).mean())
            base = float(max(np.mean(yt == 0), np.mean(yt == 1)))
            rows.append({"pair": TYPES[a] + " vs " + TYPES[b],
                         "n_samples": int(m.sum()), "n_dyads": int(nd),
                         "accuracy": round(acc, 4),
                         "majority_baseline": round(base, 4),
                         "balanced_accuracy": round(
                             float(balanced_accuracy_score(yt, yp)), 4),
                         "delta_above_baseline": round(acc - base, 4)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gender-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-iterations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perm-seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "permutation_mappings").mkdir(exist_ok=True)
    dd = Path(args.data_dir)
    gt = {int(k): v for k, v in
          json.load(open(args.gender_json))["dyad_type_analysis_set"].items()}

    rows = []
    for r in csv.DictReader(open(args.manifest)):
        if str(r.get("flagged", "")).lower() == "true":
            continue
        p = dd / r["output_npy"]
        if p.exists() and int(r["dyad_id"]) in gt:
            rows.append({"npy": r["output_npy"], "path": p, "dyad": int(r["dyad_id"]),
                         "type": gt[int(r["dyad_id"])]})
    rows.sort(key=lambda e: (e["dyad"], e["npy"]))
    t2i = {t: i for i, t in enumerate(TYPES)}

    blocks, ys, gs, meta, cur = [], [], [], [], 0
    for e in rows:
        X = to_tc(np.load(e["path"]).astype(np.float32)); n = X.shape[0]
        blocks.append(X); ys.append(np.full(n, t2i[e["type"]], np.int64))
        gs.append(np.full(n, e["dyad"], np.int64))
        meta.append({"npy": e["npy"], "dyad": e["dyad"], "type": e["type"],
                     "cls": t2i[e["type"]], "start": cur, "end": cur + n})
        cur += n
    X = np.concatenate(blocks); y = np.concatenate(ys); g = np.concatenate(gs)
    del blocks
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True); sd[sd == 0] = 1
    X -= mu; X /= sd; X = np.ascontiguousarray(X, np.float32)
    dyad_type = {e["dyad"]: e["type"] for e in rows}
    cc = np.bincount(y, minlength=3)
    per_type = {t: sorted(d for d, v in dyad_type.items() if v == t) for t in TYPES}
    print("[gender] X=%s files=%d dyads=%d classes MM/FF/MF=%s chance=%.4f"
          % (X.shape, len(rows), len(dyad_type), cc.tolist(), cc.max() / len(y)),
          flush=True)
    for t in TYPES:
        print("   %s: %d dyads %s" % (t, len(per_type[t]), per_type[t]), flush=True)

    with open(out / "sample_metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["npy", "dyad", "type", "cls", "start", "end"])
        w.writeheader(); w.writerows(meta)
    json.dump({"n_files": len(rows), "n_dyads": len(dyad_type),
               "n_samples": int(len(X)), "class_counts_MM_FF_MF": cc.tolist(),
               "majority_chance": float(cc.max() / len(y)),
               "dyads_per_type": per_type,
               "cebra_config": dict(CEBRA_KW, max_iterations=args.max_iterations,
                                    seed=args.seed)},
              open(out / "dataset_provenance.json", "w"), indent=2)

    records = []
    rd = out / "real"
    if not (rd / "metrics.json").exists():
        rd.mkdir(exist_ok=True)
        t0 = time.time()
        np.random.seed(args.seed); torch.manual_seed(args.seed)
        model = CEBRA(max_iterations=args.max_iterations, **CEBRA_KW)
        model.fit(X, y); model.save(str(rd / "model.pt"))
        emb = model.transform(X).astype(np.float32)
        np.save(rd / "embedding.npy", emb)
        lh = list(getattr(model, "state_dict_", {}).get("loss", []))
        fl = float(lh[-1]) if lh else None
        try:
            gof = float(cmetrics.goodness_of_fit_score(model, X, y))
        except Exception:
            gof = None
        m, sub, folds, ll = evaluate(emb, y, g, args.seed)
        m.update({"run": "real", "final_loss": fl, "goodness_of_fit_bits": gof,
                  "train_seconds": round(time.time() - t0, 1)})
        m["pairwise"] = pairwise(emb, y, g, args.seed, sub)
        json.dump(m, open(rd / "metrics.json", "w"), indent=2)
        json.dump(folds, open(rd / "per_fold.json", "w"), indent=2)
        np.save(rd / "eval_indices.npy", sub)
        np.save(rd / "lonlat_radius.npy", np.c_[ll[0], ll[1], ll[2]])
        try:
            ax = cebra.plot_loss(model)
            ax.get_figure().savefig(rd / "loss.png", dpi=300, facecolor="white",
                                    bbox_inches="tight")
            plt.close("all")
        except Exception:
            pass
        print("  REAL loss=%.4f GoF=%.4f LODO=%.4f bal=%.4f F1=%.4f sil=%.4f (%.1f min)"
              % (fl, gof, m["lodo_acc"], m["lodo_balanced"], m["lodo_macro_f1"],
                 m["silhouette"], (time.time() - t0) / 60), flush=True)
        del emb, model
    records.append(json.load(open(rd / "metrics.json")))

    dyads = sorted(dyad_type)
    labels = np.array([t2i[dyad_type[d]] for d in dyads])
    for ps in args.perm_seeds:
        pdir = out / ("perm%d" % ps)
        if (pdir / "metrics.json").exists():
            print("  SKIP perm%d" % ps, flush=True)
            records.append(json.load(open(pdir / "metrics.json"))); continue
        pdir.mkdir(parents=True, exist_ok=True)
        perm = np.random.RandomState(7000 + ps).permutation(len(labels))
        pmap = {d: int(labels[perm[i]]) for i, d in enumerate(dyads)}
        assert sorted(pmap.values()) == sorted(labels.tolist())
        yq = np.empty_like(y)
        for e in meta:
            yq[e["start"]:e["end"]] = pmap[e["dyad"]]
        assert np.bincount(yq, minlength=3).tolist() == cc.tolist()
        nmv = sum(1 for d in pmap if pmap[d] != t2i[dyad_type[d]])
        json.dump({"perm_seed": ps, "n_dyads_relabelled": nmv,
                   "permuted": {str(d): TYPES[pmap[d]] for d in dyads},
                   "real": {str(d): dyad_type[d] for d in dyads}},
                  open(out / "permutation_mappings" / ("perm%d.json" % ps), "w"), indent=2)
        print("\n=== GENDER PERMUTATION %d (%d/%d dyads relabelled) ==="
              % (ps, nmv, len(dyads)), flush=True)
        t0 = time.time()
        np.random.seed(args.seed); torch.manual_seed(args.seed)
        model = CEBRA(max_iterations=args.max_iterations, **CEBRA_KW)
        model.fit(X, yq); model.save(str(pdir / "model.pt"))
        emb = model.transform(X).astype(np.float32)
        lh = list(getattr(model, "state_dict_", {}).get("loss", []))
        fl = float(lh[-1]) if lh else None
        try:
            gof = float(cmetrics.goodness_of_fit_score(model, X, yq))
        except Exception:
            gof = None
        m, sub, folds, ll = evaluate(emb, yq, g, args.seed)
        m.update({"run": "perm%d" % ps, "perm_seed": ps, "final_loss": fl,
                  "goodness_of_fit_bits": gof, "n_dyads_relabelled": nmv,
                  "train_seconds": round(time.time() - t0, 1)})
        json.dump(m, open(pdir / "metrics.json", "w"), indent=2)
        np.save(pdir / "embedding_subsampled.npy", emb[sub])
        records.append(m)
        print("  perm%d: loss=%.4f GoF=%.4f LODO=%.4f bal=%.4f sil=%.4f (%.1f min)"
              % (ps, fl, gof, m["lodo_acc"], m["lodo_balanced"], m["silhouette"],
                 (time.time() - t0) / 60), flush=True)
        del emb, model

    cols = ["run", "final_loss", "goodness_of_fit_bits", "lodo_acc", "lodo_balanced",
            "lodo_macro_f1", "random_split_acc", "majority_chance", "silhouette",
            "n_dyads_relabelled", "train_seconds"]
    with open(out / "gender_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(records)
    print("\nGENDER_CEBRA_DONE")


if __name__ == "__main__":
    main()
