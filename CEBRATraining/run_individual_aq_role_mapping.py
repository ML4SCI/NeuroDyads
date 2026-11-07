#!/usr/bin/env python
"""
scripts/run_individual_aq_role_mapping.py
=========================================
Mentor item 4 -- CEBRA-map INDIVIDUAL PARTICIPANTS by their own AQ level, with
speaker and listener recordings kept separate, then ask how similar a given
person's two role recordings are to each other in the lat/long map.

Difference from everything else in this project: the unit is the *participant*,
not the dyad, and the label is that participant's own AQ-10 score (0-9 in this
cohort), not a between-person difference. Each participant contributes two
recordings -- one speaking, one listening -- which are never concatenated.

Two questions are answered:

  (a) Can AQ level be decoded across participants?
      Grouped leave-one-PARTICIPANT-out k-NN, so a person's own data is never in
      the training set. Reported for exact AQ level and for a coarse binning,
      because several AQ levels have very few people.

  (b) Is a person's speaking brain-state closer to their own listening state
      than to other people's?
      For each participant we take the centroid of their speak embedding and the
      centroid of their listen embedding, and compute cosine similarity (the
      embedding lives on a sphere -- CEBRA was trained with cosine distance).
      The within-participant value is compared against a null built from
      speak(p) vs listen(q) for every p != q. This is the "within-individual
      neural similarity for roles" comparison.

Caveat carried in from the rest of the project: AQ level is constant within a
participant, exactly as AQ magnitude was constant within a dyad, so decoding it
across participants is open to the same identity confound. A participant-level
permutation control is therefore run alongside.
"""
from __future__ import annotations

import argparse, csv, json, re, sys, time, warnings
from pathlib import Path

import numpy as np

try:
    import torch, cebra
    from cebra import CEBRA
    from cebra.integrations.sklearn import metrics as cmetrics
except ImportError:
    sys.exit("need torch + cebra")
import mne
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, balanced_accuracy_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

mne.set_log_level("ERROR"); warnings.filterwarnings("ignore")

CEBRA_KW = dict(model_architecture="offset10-model", batch_size=512, learning_rate=3e-4,
                temperature=1.12, conditional="time_delta", output_dimension=3,
                distance="cosine", device="cuda_if_available", verbose=True, time_offsets=10)
FILE_RE = re.compile(r"^(?:cut_)?dyad_?(\d+)_(\d+)_(speak|listen)\.edf$", re.I)
FAULTY = {(38, 98), (41, 104)}
SUB = 60000


def wrap180(x):
    return (x + 180) % 360 - 180


def eeg_picks(names):
    return [i for i, c in enumerate(names)
            if not any(t in c.upper() for t in ("VREF", "STATUS", "TRIGGER", "STI"))]


def load_participant_aq(xlsx, sheet="Main"):
    """participant id -> AQ score, from either side of each pair row."""
    import pandas as pd
    df = pd.read_excel(xlsx, sheet_name=sheet)
    df = df[df["dyad_id"].notna()]
    aq = {}
    for _, r in df.iterrows():
        for idc, aqc in (("Speaker ID", "Speaker AQ"), ("Listener ID", "Listener AQ")):
            try:
                pid = int(r[idc]); val = float(r[aqc])
            except Exception:
                continue
            if val == val:
                aq.setdefault(pid, val)
    return aq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--aq-sheet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-iterations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-perms", type=int, default=200,
                    help="participant-level label permutations (decoder-level)")
    ap.add_argument("--exclude-flagged", action="store_true", default=True)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    aq_map = load_participant_aq(args.aq_sheet)

    files = []
    for p in sorted(Path(args.input_dir).glob("*.edf")):
        m = FILE_RE.match(p.name)
        if not m:
            continue
        d, pid, role = int(m.group(1)), int(m.group(2)), m.group(3).lower()
        if args.exclude_flagged and (d, pid) in FAULTY:
            print(f"  excluding flagged {p.name}"); continue
        if pid not in aq_map:
            print(f"  no AQ for participant {pid} ({p.name})"); continue
        files.append({"path": p, "dyad": d, "pid": pid, "role": role, "aq": aq_map[pid]})
    if not files:
        sys.exit("no usable recordings")

    # keep only participants that have BOTH roles -- (b) needs the pair
    have = {}
    for f in files:
        have.setdefault(f["pid"], set()).add(f["role"])
    both = {p for p, r in have.items() if {"speak", "listen"} <= r}
    files = [f for f in files if f["pid"] in both]
    print(f"{len(files)} recordings, {len(both)} participants with both roles", flush=True)

    crop = min(mne.io.read_raw_edf(str(f["path"]), preload=False,
                                   verbose="ERROR").n_times for f in files)
    print(f"global minimum crop = {crop} samples", flush=True)

    blocks, meta, cur = [], [], 0
    for f in files:
        raw = mne.io.read_raw_edf(str(f["path"]), preload=True, verbose="ERROR")
        X = raw.get_data()[eeg_picks(raw.ch_names)][:, :crop].T.astype(np.float32)
        raw.close()
        blocks.append(X)
        meta.append({**{k: f[k] for k in ("dyad", "pid", "role", "aq")},
                     "file": f["path"].name, "start": cur, "end": cur + X.shape[0]})
        cur += X.shape[0]
    X = np.concatenate(blocks); del blocks
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True); sd[sd == 0] = 1
    X -= mu; X /= sd; X = np.ascontiguousarray(X, np.float32)

    aq_vals = sorted({m["aq"] for m in meta})
    aq_to_cls = {v: i for i, v in enumerate(aq_vals)}
    y = np.zeros(len(X), np.int64); pidv = np.zeros(len(X), np.int64)
    rolev = np.zeros(len(X), np.int64); aqv = np.zeros(len(X), np.float32)
    for m in meta:
        s, e = m["start"], m["end"]
        y[s:e] = aq_to_cls[m["aq"]]; pidv[s:e] = m["pid"]
        rolev[s:e] = 0 if m["role"] == "speak" else 1; aqv[s:e] = m["aq"]
    print(f"X={X.shape}  AQ levels present={aq_vals}  "
          f"participants={len(both)}", flush=True)

    ppl = {}
    for m in meta:
        ppl[m["pid"]] = m["aq"]
    import collections
    aq_people = collections.Counter(ppl.values())
    print(f"people per AQ level: {dict(sorted(aq_people.items()))}", flush=True)

    json.dump({"n_recordings": len(meta), "n_participants": len(both),
               "crop_samples": int(crop), "n_samples": int(len(X)),
               "aq_levels_present": aq_vals,
               "people_per_aq_level": {str(k): v for k, v in sorted(aq_people.items())},
               "cebra_config": {**CEBRA_KW, "max_iterations": args.max_iterations,
                                "seed": args.seed},
               "recordings": [{k: (str(v) if k == "path" else v) for k, v in m.items()}
                              for m in meta]},
              open(out / "dataset_provenance.json", "w"), indent=2)

    # ---------------- train ----------------
    t0 = time.time()
    model = CEBRA(max_iterations=args.max_iterations, **CEBRA_KW)
    model.fit(X, y)
    model.save(str(out / "model.pt"))
    emb = model.transform(X).astype(np.float32)
    np.save(out / "embedding.npy", emb)
    loss_hist = list(getattr(model, "state_dict_", {}).get("loss", []))
    final_loss = float(loss_hist[-1]) if loss_hist else None
    try:
        gof = float(cmetrics.goodness_of_fit_score(model, X, y))
    except Exception:
        gof = None
    print(f"trained in {(time.time()-t0)/60:.1f} min  loss={final_loss:.4f} GoF={gof}",
          flush=True)
    with open(out / "sample_metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "dyad", "pid", "role", "aq", "start", "end"])
        w.writeheader(); w.writerows(meta)
    try:
        ax = cebra.plot_loss(model); ax.get_figure().savefig(out / "loss.png", dpi=300,
            facecolor="white", bbox_inches="tight"); plt.close("all")
    except Exception:
        pass

    # ---------------- (a) decode AQ level, leave-one-participant-out -------------
    rng = np.random.RandomState(args.seed)
    sub = rng.choice(len(X), min(SUB, len(X)), replace=False); sub.sort()
    E, Y, P, R, A = emb[sub], y[sub], pidv[sub], rolev[sub], aqv[sub]
    np.save(out / "eval_indices.npy", sub)

    def lopo(feat, lab, grp, k=5):
        yt, yp = [], []
        for q in sorted(set(grp.tolist())):
            te = grp == q; tr = ~te
            if te.sum() == 0 or len(set(lab[tr].tolist())) < 2:
                continue
            sc = StandardScaler().fit(feat[tr])
            m = KNeighborsClassifier(k).fit(sc.transform(feat[tr]), lab[tr])
            yp.append(m.predict(sc.transform(feat[te]))); yt.append(lab[te])
        return np.concatenate(yt), np.concatenate(yp)

    yt, yp = lopo(E, Y, P)
    acc_exact = float((yt == yp).mean())
    bal_exact = float(balanced_accuracy_score(yt, yp))
    chance_exact = float(np.bincount(Y).max() / len(Y))

    # coarse AQ binning -- several exact levels have only 1-2 people
    med = float(np.median(sorted(ppl.values())))
    Ybin = (A > med).astype(np.int64)
    ytb, ypb = lopo(E, Ybin, P)
    acc_bin = float((ytb == ypb).mean())
    bal_bin = float(balanced_accuracy_score(ytb, ypb))
    chance_bin = float(np.bincount(Ybin).max() / len(Ybin))

    # role decoding, as a reference point that is NOT identity-confounded
    ytr, ypr = lopo(E, R, P)
    acc_role = float((ytr == ypr).mean())
    chance_role = float(np.bincount(R).max() / len(R))

    # participant-level permutation of AQ (decoder-level control)
    people = sorted(ppl)
    lab_by_person = np.array([aq_to_cls[ppl[q]] for q in people])
    null = []
    for i in range(args.n_perms):
        perm = np.random.RandomState(5000 + i).permutation(len(people))
        pm = {q: int(lab_by_person[perm[j]]) for j, q in enumerate(people)}
        yq = np.array([pm[int(q)] for q in P])
        a, b = lopo(E, yq, P)
        null.append(float((a == b).mean()))
    null = np.array(null)
    p_exact = (1 + int((null >= acc_exact).sum())) / (1 + len(null))

    # ---------------- (b) within-individual role similarity ----------------
    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    cent = {}
    for m in meta:
        cent[(m["pid"], m["role"])] = unit(emb[m["start"]:m["end"]].mean(0))
    within, across = [], []
    rows = []
    for q in sorted(both):
        s, l = cent.get((q, "speak")), cent.get((q, "listen"))
        if s is None or l is None:
            continue
        w = float(np.dot(s, l))
        within.append(w)
        oth = [float(np.dot(s, cent[(r, "listen")])) for r in both
               if r != q and (r, "listen") in cent]
        across += oth
        rows.append({"participant": q, "aq": ppl[q],
                     "within_role_cos": round(w, 4),
                     "mean_across_person_cos": round(float(np.mean(oth)), 4) if oth else None,
                     "rank_of_own_listener": int(1 + sum(o > w for o in oth)),
                     "n_compared": len(oth)})
    within = np.array(within); across = np.array(across)
    u = mannwhitneyu(within, across, alternative="greater")
    top1 = float(np.mean([r["rank_of_own_listener"] == 1 for r in rows]))
    with open(out / "within_individual_role_similarity.csv", "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w_.writeheader(); w_.writerows(rows)

    # ---------------- lat/long ----------------
    Ec = E - np.median(E, axis=0)
    x_, y_, z_ = Ec[:, 0], Ec[:, 1], Ec[:, 2]
    r_ = np.sqrt(x_**2 + y_**2 + z_**2)
    lon = np.degrees(np.arctan2(y_, x_)); lat = np.degrees(np.arctan2(z_, np.sqrt(x_**2 + y_**2)))
    cm = np.degrees(np.arctan2(np.sin(np.radians(lon)).mean(), np.cos(np.radians(lon)).mean()))
    lon_r = wrap180(lon - cm)
    np.save(out / "lonlat_radius.npy", np.c_[lon_r, lat, r_])
    ks_role = {c: float(ks_2samp(v[R == 0], v[R == 1]).statistic)
               for c, v in [("latitude", lat), ("longitude", lon_r), ("radius", r_)]}
    ks_aq = {c: float(ks_2samp(v[Ybin == 0], v[Ybin == 1]).statistic)
             for c, v in [("latitude", lat), ("longitude", lon_r), ("radius", r_)]}
    try:
        s2 = rng.choice(len(E), min(10000, len(E)), replace=False)
        sil_aq = float(silhouette_score(E[s2], Y[s2]))
        sil_role = float(silhouette_score(E[s2], R[s2]))
    except Exception:
        sil_aq = sil_role = None

    metrics = {"analysis": "individual_aq_role", "n_participants": len(both),
               "n_recordings": len(meta), "n_samples": int(len(X)),
               "aq_levels_present": aq_vals, "final_loss": final_loss,
               "goodness_of_fit_bits": gof,
               "decode_exact_aq_lopo_knn5": acc_exact,
               "decode_exact_aq_balanced": bal_exact,
               "decode_exact_aq_chance": chance_exact,
               "decode_exact_aq_perm_null_mean": float(null.mean()),
               "decode_exact_aq_perm_null_std": float(null.std()),
               "decode_exact_aq_empirical_p": p_exact,
               "decode_binary_aq_lopo_knn5": acc_bin,
               "decode_binary_aq_balanced": bal_bin,
               "decode_binary_aq_chance": chance_bin,
               "aq_median_split_at": med,
               "decode_role_lopo_knn5": acc_role, "decode_role_chance": chance_role,
               "silhouette_aq": sil_aq, "silhouette_role": sil_role,
               "ks_by_role": ks_role, "ks_by_aq_binary": ks_aq,
               "within_role_cos_mean": float(within.mean()),
               "within_role_cos_std": float(within.std()),
               "across_person_cos_mean": float(across.mean()),
               "across_person_cos_std": float(across.std()),
               "mannwhitney_U": float(u.statistic), "mannwhitney_p": float(u.pvalue),
               "own_listener_ranked_first_frac": top1,
               "cebra_version": cebra.__version__}
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)

    # ---------------- figures ----------------
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    sc0 = axs[0].scatter(lon_r, lat, s=3, alpha=0.3, c=A, cmap="viridis", linewidths=0)
    axs[0].set_xlabel("Longitude"); axs[0].set_ylabel("Latitude")
    axs[0].set_title("Individual participants coloured by their own AQ level", fontsize=10)
    fig.colorbar(sc0, ax=axs[0], label="AQ-10 score")
    for rv, col, lb in [(0, "#0173B2", "speaking"), (1, "#DE8F05", "listening")]:
        m = R == rv
        axs[1].scatter(lon_r[m], lat[m], s=3, alpha=0.3, color=col, label=lb, linewidths=0)
    axs[1].set_xlabel("Longitude"); axs[1].set_ylabel("Latitude")
    axs[1].set_title(f"Same map coloured by role "
                     f"(role LOPO decoding {acc_role:.3f}, chance {chance_role:.3f})",
                     fontsize=10)
    axs[1].legend(markerscale=5)
    fig.tight_layout(); fig.savefig(out / "individual_aq_lonlat.png", dpi=300,
                                    facecolor="white", bbox_inches="tight"); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    axs[0].hist(across, bins=40, alpha=0.75, color="#BBBBBB", density=True,
                label=f"different people (n={len(across)})")
    axs[0].hist(within, bins=15, alpha=0.8, color="#D55E00", density=True,
                label=f"same person, both roles (n={len(within)})")
    axs[0].axvline(within.mean(), color="#D55E00", lw=2)
    axs[0].axvline(across.mean(), color="#555555", lw=2, ls="--")
    axs[0].set_xlabel("cosine similarity of embedding centroids")
    axs[0].set_ylabel("density")
    axs[0].set_title(f"Within-individual role similarity\n"
                     f"within {within.mean():.3f} vs across {across.mean():.3f}, "
                     f"Mann-Whitney p={u.pvalue:.3g}", fontsize=9)
    axs[0].legend(fontsize=8)
    axs[1].scatter([r["aq"] for r in rows], [r["within_role_cos"] for r in rows],
                   s=45, color="#0173B2")
    axs[1].axhline(across.mean(), color="#555555", ls="--", lw=1.5,
                   label="mean across-person")
    axs[1].set_xlabel("participant AQ-10 score")
    axs[1].set_ylabel("speak-vs-listen cosine similarity")
    axs[1].set_title("Does role consistency depend on AQ?", fontsize=10)
    axs[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "individual_role_similarity.png", dpi=300,
                                    facecolor="white", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(null, bins=30, color="#9ecae1", edgecolor="white",
            label=f"participant-permuted null (n={len(null)})")
    ax.axvline(acc_exact, color="#D55E00", lw=2.5, label=f"real = {acc_exact:.3f}")
    ax.axvline(chance_exact, color="#999999", ls=":", lw=1.5,
               label=f"majority chance = {chance_exact:.3f}")
    ax.set_xlabel("leave-one-participant-out 5-NN accuracy (exact AQ level)")
    ax.set_ylabel("count")
    ax.set_title(f"AQ-level decoding vs participant-level permutation, p={p_exact:.3f}",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "individual_aq_permutation_null.png", dpi=300,
                                    facecolor="white", bbox_inches="tight"); plt.close(fig)

    print(f"\n[individual] exact-AQ LOPO 5-NN={acc_exact:.4f} (chance {chance_exact:.4f}, "
          f"perm null {null.mean():.4f}+-{null.std():.4f}, p={p_exact:.3f})")
    print(f"[individual] binary-AQ LOPO 5-NN={acc_bin:.4f} (chance {chance_bin:.4f})")
    print(f"[individual] ROLE LOPO 5-NN={acc_role:.4f} (chance {chance_role:.4f})")
    print(f"[individual] within-person role cos={within.mean():.4f}+-{within.std():.4f} vs "
          f"across-person {across.mean():.4f}+-{across.std():.4f}, "
          f"MWU p={u.pvalue:.3g}, own-listener-ranked-first {top1*100:.1f}%")
    print("INDIVIDUAL_AQ_ROLE_DONE")


if __name__ == "__main__":
    main()
