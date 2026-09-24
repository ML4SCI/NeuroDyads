#!/usr/bin/env python
"""
scripts/run_dyad_aq_permutation_control.py
==========================================
TASK 4 (highest priority) -- DYAD-LEVEL AQ PERMUTATION CONTROL.

The question: is the speaker-first AQ-magnitude decoding real AQ structure, or
could ANY random assignment of Low/High to whole dyads be decoded just as well
(i.e. the model is really reading dyad identity)?

Design (the strong version -- CEBRA is RETRAINED per permutation):
  1. Load the validated FULL-BAND speaker-first dataset (33 dyads x 2 files).
  2. For permutation seed s: permute the dyad -> AQ-magnitude assignment.
     * both speaker-first files of a dyad get the SAME permuted label
     * the number of Low/High DYADS is preserved exactly (a pure relabelling
       of which dyads are Low vs High), so per-class sample counts are
       identical to the real run
     * sample ordering and file structure are untouched
  3. Retrain CEBRA supervised on the permuted label with the SAME config and the
     SAME model init seed (0) as the real run, so the only thing that changes is
     the label.
  4. Recompute the full metric suite against the permuted label.
  5. Compare the real run to the null distribution.

This is NOT post-hoc decoder label shuffling -- the encoder itself is retrained.
A separate cheap dyad-level decoder permutation test (>=1000 perms on the fixed
real embedding) is run by run_posthoc_decoder_permutation.py and is clearly
labelled a secondary control.

Config is copied verbatim from scripts/run_band_cebra_analysis.py so real and
null are directly comparable.
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
    import sys; sys.exit("ERROR: need torch + cebra")
from scipy.stats import ks_2samp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture

# ---- CEBRA config: MUST match the real full-band speaker-first run ----
CEBRA_KW = dict(model_architecture="offset10-model", batch_size=512, learning_rate=3e-4,
                temperature=1.12, conditional="time_delta", output_dimension=3,
                distance="cosine", device="cuda_if_available", verbose=True, time_offsets=10)
SUBSAMPLE = 60000


def to_tc(x):
    return x.T if x.shape[0] < x.shape[1] else x


def wrap180(x):
    return (x + 180) % 360 - 180


def load_manifest(mpath, data_dir, exclude_flagged=True):
    rows = []
    for r in csv.DictReader(open(mpath)):
        if exclude_flagged and str(r.get("flagged", "")).lower() == "true":
            continue
        p = data_dir / r["output_npy"]
        if p.exists():
            rows.append({"npy": r["output_npy"], "path": p, "dyad": int(r["dyad_id"]),
                         "mag": int(r["aq_magnitude"]), "abs_daq": float(r["abs_daq"])})
    rows.sort(key=lambda e: (e["dyad"], e["npy"]))
    return rows


def fixed_grouped_knn(X, y, g, k=5):
    """The exact holdout used by run_band_cebra_analysis.py (every 3rd dyad held out).
    Kept so the null is comparable to the already-published real number."""
    gs = sorted(set(g.tolist())); te = set(gs[::3])
    tr = np.where(~np.isin(g, list(te)))[0]; te_i = np.where(np.isin(g, list(te)))[0]
    if not len(tr) or not len(te_i):
        return None
    sc = StandardScaler().fit(X[tr])
    return float(KNeighborsClassifier(k).fit(sc.transform(X[tr]), y[tr]).score(sc.transform(X[te_i]), y[te_i]))


def leave_one_dyad_out_knn(X, y, g, k=5):
    """Primary decoder: full leave-one-dyad-out CV, pooled accuracy."""
    dy = sorted(set(g.tolist()))
    correct = tot = 0
    per_fold = []
    for d in dy:
        te = g == d; tr = ~te
        if te.sum() == 0 or tr.sum() == 0 or len(set(y[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        m = KNeighborsClassifier(k).fit(sc.transform(X[tr]), y[tr])
        pred = m.predict(sc.transform(X[te]))
        c = int((pred == y[te]).sum()); correct += c; tot += int(te.sum())
        per_fold.append({"dyad": int(d), "n": int(te.sum()), "acc": c / int(te.sum())})
    return (correct / tot if tot else None), per_fold


def random_split_knn(X, y, seed, k=5):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X)); cut = int(0.7 * len(X))
    tr, te = idx[:cut], idx[cut:]
    sc = StandardScaler().fit(X[tr])
    return float(KNeighborsClassifier(k).fit(sc.transform(X[tr]), y[tr]).score(sc.transform(X[te]), y[te]))


def lonlat(E):
    Ec = E - np.median(E, axis=0)
    x_, y_, z_ = Ec[:, 0], Ec[:, 1], Ec[:, 2]
    r = np.sqrt(x_**2 + y_**2 + z_**2)
    lon = np.degrees(np.arctan2(y_, x_))
    lat = np.degrees(np.arctan2(z_, np.sqrt(x_**2 + y_**2)))
    cm = np.degrees(np.arctan2(np.sin(np.radians(lon)).mean(), np.cos(np.radians(lon)).mean()))
    return wrap180(lon - cm), lat, r, float(cm)


def evaluate(emb, y, g, seed, kmax=6):
    """Full metric suite on a 3D embedding given labels y and dyad groups g."""
    rng = np.random.RandomState(seed)
    T = len(emb)
    sub = rng.choice(T, min(SUBSAMPLE, T), replace=False)
    E, Y, G = emb[sub], y[sub], g[sub]
    out = {}
    out["knn5_grouped_fixed"] = fixed_grouped_knn(E, Y, G)
    lodo, per_fold = leave_one_dyad_out_knn(E, Y, G)
    out["knn5_leave_one_dyad_out"] = lodo
    out["knn5_random_split"] = random_split_knn(E, Y, seed)
    out["majority_chance"] = float(np.bincount(Y).max() / len(Y))
    try:
        s2 = rng.choice(len(sub), min(10000, len(sub)), replace=False)
        out["silhouette"] = float(silhouette_score(E[s2], Y[s2])) if len(set(Y[s2].tolist())) > 1 else None
    except Exception:
        out["silhouette"] = None
    lon_r, lat, r, cm = lonlat(E)
    out["lon_rotation_deg"] = cm
    if len(set(Y.tolist())) > 1:
        out["ks_latitude"] = float(ks_2samp(lat[Y == 0], lat[Y == 1]).statistic)
        out["ks_longitude"] = float(ks_2samp(lon_r[Y == 0], lon_r[Y == 1]).statistic)
        out["ks_radius"] = float(ks_2samp(r[Y == 0], r[Y == 1]).statistic)
    # label-agnostic GMM on (lon, lat)
    feat = np.c_[lon_r, lat]
    bics = []
    models = {}
    for k in range(1, kmax + 1):
        gm = GaussianMixture(k, covariance_type="full", random_state=seed, n_init=2, max_iter=300).fit(feat)
        bics.append(float(gm.bic(feat))); models[k] = gm
    out["gmm_bic_by_k"] = {k: round(b, 1) for k, b in zip(range(1, kmax + 1), bics)}
    out["gmm_bestK_by_bic"] = int(np.argmin(bics) + 1)
    cl2 = models[2].predict(feat)
    out["gmm_K2_ARI"] = float(adjusted_rand_score(Y, cl2))
    out["gmm_K2_purity"] = float(sum(np.bincount(Y[cl2 == c]).max() for c in np.unique(cl2)) / len(Y))
    return out, sub, per_fold


def permute_dyad_labels(dyad_to_label, seed):
    """Permute WHICH dyads carry which label. Preserves the multiset of dyad
    labels exactly, hence per-class file and sample counts are unchanged."""
    dyads = sorted(dyad_to_label)
    labels = np.array([dyad_to_label[d] for d in dyads])
    rng = np.random.RandomState(1000 + seed)
    perm = rng.permutation(len(labels))
    return {d: int(labels[perm[i]]) for i, d in enumerate(dyads)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--condition", default="fullband_speakerfirst")
    ap.add_argument("--max-iterations", type=int, default=5000)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--perm-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--real-embedding", default=None,
                    help="existing real embedding.npy -- re-scored with identical code")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "permutation_mappings").mkdir(exist_ok=True)
    data_dir = Path(args.data_dir)

    rows = load_manifest(args.manifest, data_dir)
    if not rows:
        import sys; sys.exit("no files after filtering")

    # ---- build dataset once, reuse for every permutation ----
    blocks, ys, gs, meta = [], [], [], []
    cur = 0
    for e in rows:
        Xf = to_tc(np.load(e["path"]).astype(np.float32))
        n = Xf.shape[0]
        blocks.append(Xf); ys.append(np.full(n, e["mag"], np.int64)); gs.append(np.full(n, e["dyad"], np.int64))
        meta.append({"npy": e["npy"], "dyad": e["dyad"], "mag": e["mag"], "start": cur, "end": cur + n})
        cur += n
    X = np.concatenate(blocks); y_real = np.concatenate(ys); g = np.concatenate(gs)
    del blocks
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True); sd[sd == 0] = 1
    X -= mu; X /= sd
    X = np.ascontiguousarray(X, dtype=np.float32)
    T = X.shape[0]
    dyad_to_label = {e["dyad"]: e["mag"] for e in rows}
    n_dyads = len(dyad_to_label)
    print(f"[perm-control] X={X.shape} files={len(rows)} dyads={n_dyads} "
          f"classes={np.bincount(y_real).tolist()} chance={np.bincount(y_real).max()/T:.4f}", flush=True)

    with open(out / "dataset_provenance.json", "w") as f:
        json.dump({"condition": args.condition, "data_dir": str(data_dir),
                   "manifest": str(args.manifest), "n_files": len(rows), "n_dyads": n_dyads,
                   "n_samples": int(T), "n_channels": int(X.shape[1]),
                   "class_counts": np.bincount(y_real).tolist(),
                   "majority_chance": float(np.bincount(y_real).max() / T),
                   "real_dyad_to_label": {str(k): v for k, v in sorted(dyad_to_label.items())},
                   "cebra_config": {**CEBRA_KW, "max_iterations": args.max_iterations,
                                    "model_seed": args.model_seed},
                   "files": [e["npy"] for e in rows]}, f, indent=2)

    records = []

    # ---- REAL reference: re-score the existing embedding with identical code ----
    if args.real_embedding and Path(args.real_embedding).exists():
        print("[perm-control] scoring REAL embedding with identical evaluation code", flush=True)
        emb_real = np.load(args.real_embedding).astype(np.float32)
        if len(emb_real) == T:
            m, sub, per_fold = evaluate(emb_real, y_real, g, args.model_seed)
            m.update({"run": "real", "perm_seed": None, "final_loss": None,
                      "goodness_of_fit_bits": None})
            json.dump({"metrics": m, "per_fold_lodo": per_fold},
                      open(out / "real_rescored_metrics.json", "w"), indent=2)
            np.save(out / "real_eval_indices.npy", sub)
            records.append(m)
            print(f"  REAL  LODO 5-NN={m['knn5_leave_one_dyad_out']:.4f} "
                  f"fixed={m['knn5_grouped_fixed']:.4f} sil={m['silhouette']}", flush=True)
        else:
            print(f"  !! real embedding length {len(emb_real)} != {T}; skipping re-score", flush=True)
        del emb_real

    # ---- PERMUTATIONS ----
    for ps in args.perm_seeds:
        pdir = out / f"perm{ps}"
        if (pdir / "metrics.json").exists():
            print(f"[perm-control] SKIP perm{ps} (already complete)", flush=True)
            records.append(json.load(open(pdir / "metrics.json")))
            continue
        pdir.mkdir(parents=True, exist_ok=True)
        pmap = permute_dyad_labels(dyad_to_label, ps)
        # sanity: label multiset preserved
        assert sorted(pmap.values()) == sorted(dyad_to_label.values()), "class counts changed!"
        y_perm = np.empty_like(y_real)
        for e_meta in meta:
            y_perm[e_meta["start"]:e_meta["end"]] = pmap[e_meta["dyad"]]
        assert np.bincount(y_perm).tolist() == np.bincount(y_real).tolist(), "sample counts changed!"
        n_moved = sum(1 for d in pmap if pmap[d] != dyad_to_label[d])
        json.dump({"perm_seed": ps,
                   "real_dyad_to_label": {str(k): v for k, v in sorted(dyad_to_label.items())},
                   "permuted_dyad_to_label": {str(k): v for k, v in sorted(pmap.items())},
                   "n_dyads_with_changed_label": n_moved,
                   "class_counts": np.bincount(y_perm).tolist()},
                  open(out / "permutation_mappings" / f"perm{ps}_mapping.json", "w"), indent=2)

        print(f"\n=== PERMUTATION {ps} ({n_moved}/{n_dyads} dyads relabelled) ===", flush=True)
        t0 = time.time()
        np.random.seed(args.model_seed); torch.manual_seed(args.model_seed)
        model = CEBRA(max_iterations=args.max_iterations, **CEBRA_KW)
        model.fit(X, y_perm)
        model.save(str(pdir / "model.pt"))
        emb = model.transform(X).astype(np.float32)
        train_s = time.time() - t0

        loss_hist = list(getattr(model, "state_dict_", {}).get("loss", []))
        final_loss = float(loss_hist[-1]) if len(loss_hist) else None
        try:
            gof = float(cmetrics.goodness_of_fit_score(model, X, y_perm))
        except Exception as ex:
            print("gof fail", ex); gof = None

        m, sub, per_fold = evaluate(emb, y_perm, g, args.model_seed)
        m.update({"run": f"perm{ps}", "perm_seed": ps, "final_loss": final_loss,
                  "goodness_of_fit_bits": gof, "train_seconds": round(train_s, 1),
                  "n_dyads_relabelled": n_moved})
        json.dump(m, open(pdir / "metrics.json", "w"), indent=2)
        json.dump(per_fold, open(pdir / "per_fold_lodo.json", "w"), indent=2)
        np.save(pdir / "eval_indices.npy", sub)
        np.save(pdir / "embedding_subsampled.npy", emb[sub])
        if loss_hist:
            np.save(pdir / "loss_history.npy", np.array(loss_hist, dtype=np.float32))
        records.append(m)
        print(f"  perm{ps}: loss={final_loss:.4f} GoF={gof} LODO5NN={m['knn5_leave_one_dyad_out']:.4f} "
              f"fixed={m['knn5_grouped_fixed']:.4f} sil={m['silhouette']} "
              f"({train_s/60:.1f} min)", flush=True)
        del emb, model

    # ---- write combined CSV ----
    cols = ["run", "perm_seed", "n_dyads_relabelled", "final_loss", "goodness_of_fit_bits",
            "knn5_leave_one_dyad_out", "knn5_grouped_fixed", "knn5_random_split",
            "majority_chance", "silhouette", "ks_latitude", "ks_longitude", "ks_radius",
            "gmm_K2_ARI", "gmm_K2_purity", "gmm_bestK_by_bic", "train_seconds"]
    with open(out / "permutation_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)
    print(f"\n[perm-control] wrote {out/'permutation_metrics.csv'}  ({len(records)} rows)", flush=True)
    print("PERMUTATION_CONTROL_DONE", flush=True)


if __name__ == "__main__":
    main()
