#!/usr/bin/env python
"""
scripts/compare_new_nonosc_upload.py
====================================
Is the newly uploaded Non-Oscillatory set actually new?

Step 1 (cheap, exact): every zip entry carries a CRC32 in its central directory.
Compare that against the CRC32 of the previously extracted file of the same name.
Matching CRC + matching size = byte-identical, no extraction needed.

Step 2 (only for files that differ): extract a sample and run the decisive
spectral test -- alpha prominence over a fitted 1/f line. A genuine aperiodic
export shows ~0 dB; unfiltered data shows the same prominence as the raw file.
"""
from __future__ import annotations

import argparse, csv, json, zipfile, zlib
from pathlib import Path

import numpy as np


def crc32_of(path, chunk=1 << 22):
    c = 0
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            c = zlib.crc32(b, c)
    return c & 0xFFFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-zip-dir", required=True)
    ap.add_argument("--old-extracted-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sample-extract", type=int, default=4)
    ap.add_argument("--work-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    oldroot = Path(args.old_extracted_dir)

    # index the old extraction by basename
    old = {}
    for p in oldroot.rglob("*.edf"):
        old[p.name] = p
    print(f"old extraction: {len(old)} EDFs in {oldroot}", flush=True)

    entries = []
    for z in sorted(Path(args.new_zip_dir).glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for i in zf.infolist():
                if i.filename.lower().endswith(".edf"):
                    entries.append({"zip": z.name, "entry": i.filename,
                                    "basename": Path(i.filename).name,
                                    "new_size": i.file_size, "new_crc": i.CRC})
    print(f"new upload: {len(entries)} EDF entries across "
          f"{len(set(e['zip'] for e in entries))} zip parts", flush=True)

    rows, n_same, n_diff, n_missing = [], 0, 0, 0
    for k, e in enumerate(entries, 1):
        b = e["basename"]
        op = old.get(b)
        if op is None:
            e.update({"old_size": None, "old_crc": None, "verdict": "NEW_FILE"})
            n_missing += 1
        else:
            os_ = op.stat().st_size
            oc = crc32_of(op)
            same = (os_ == e["new_size"]) and (oc == e["new_crc"])
            e.update({"old_size": os_, "old_crc": oc,
                      "verdict": "IDENTICAL" if same else "DIFFERENT"})
            n_same += same; n_diff += (not same)
        rows.append(e)
        if k % 40 == 0 or k == len(entries):
            print(f"  [{k}/{len(entries)}] identical={n_same} different={n_diff} "
                  f"new={n_missing}", flush=True)

    with open(out / "new_nonosc_crc_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["zip", "entry", "basename", "new_size", "old_size",
                                          "new_crc", "old_crc", "verdict"],
                           extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    verdict = ("IDENTICAL_TO_PREVIOUS" if n_diff == 0 and n_missing == 0
               else "PARTIALLY_CHANGED" if n_same else "FULLY_CHANGED")
    summary = {"n_entries": len(entries), "n_identical": n_same, "n_different": n_diff,
               "n_not_in_old": n_missing, "verdict": verdict}

    # ---------- step 2: spectral test on files that actually differ ----------
    spec_rows = []
    changed = [e for e in rows if e["verdict"] in ("DIFFERENT", "NEW_FILE")]
    if changed:
        import warnings, mne
        from scipy.signal import welch
        mne.set_log_level("ERROR"); warnings.filterwarnings("ignore")

        def prominence(path, secs=60):
            r = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
            sf = float(r.info["sfreq"])
            pk = [i for i, c in enumerate(r.ch_names)
                  if not any(t in c.upper() for t in ("VREF", "STATUS", "TRIGGER", "STI"))]
            X = r.get_data()[pk][:, :int(min(r.n_times, secs * sf))]
            r.close()
            f, P = welch(X, fs=sf, nperseg=int(4 * sf), axis=-1)
            Pm = P.mean(0)
            band = (f >= 2) & (f <= 40) & (Pm > 0)
            fitm = band & ~((f >= 7) & (f <= 14))
            co = np.polyfit(np.log10(f[fitm]), np.log10(Pm[fitm]), 1)
            am = (f >= 8) & (f <= 12)
            pred = 10 ** np.polyval(co, np.log10(f[am]))
            return float(10 * np.log10(Pm[am].mean() / pred.mean())), float(co[0]), sf

        picks = changed[:: max(1, len(changed) // args.sample_extract)][:args.sample_extract]
        for e in picks:
            zp = Path(args.new_zip_dir) / e["zip"]
            with zipfile.ZipFile(zp) as zf:
                tgt = work / e["basename"]
                with zf.open(e["entry"]) as src, open(tgt, "wb") as dst:
                    dst.write(src.read())
            newp, news, newsf = prominence(tgt)
            rec = {"basename": e["basename"], "new_alpha_prominence_db": round(newp, 3),
                   "new_1f_slope": round(news, 3), "new_sfreq": newsf}
            op = old.get(e["basename"])
            if op:
                oldp, olds, oldsf = prominence(op)
                rec.update({"old_alpha_prominence_db": round(oldp, 3),
                            "old_1f_slope": round(olds, 3), "old_sfreq": oldsf})
            spec_rows.append(rec)
            print(f"  spectral: {e['basename']}  new alpha {newp:+.2f} dB slope {news:+.3f}"
                  + (f" | old alpha {rec.get('old_alpha_prominence_db'):+.2f} dB "
                     f"slope {rec.get('old_1f_slope'):+.3f}" if op else ""), flush=True)
            try:
                tgt.unlink()
            except Exception:
                pass
        if spec_rows:
            with open(out / "new_nonosc_spectral_check.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(spec_rows[0].keys()))
                w.writeheader(); w.writerows(spec_rows)
            ap_mean = float(np.mean([r["new_alpha_prominence_db"] for r in spec_rows]))
            summary["new_alpha_prominence_mean_db"] = round(ap_mean, 3)
            summary["looks_aperiodic"] = bool(abs(ap_mean) < 0.5)

    json.dump(summary, open(out / "new_nonosc_summary.json", "w"), indent=2)

    L = ["# Newly uploaded Non-Oscillatory set — is it different?", "",
         f"New upload: `{args.new_zip_dir}`  ({len(entries)} EDF entries)  ",
         f"Compared against the previously extracted set: `{oldroot}`", "",
         "## Byte-level comparison (zip CRC32 vs extracted file CRC32)", "",
         "| outcome | files |", "|---|---|",
         f"| byte-identical to the previous upload | **{n_same}** |",
         f"| different content | **{n_diff}** |",
         f"| not present in the previous upload | **{n_missing}** |", "",
         f"**Verdict: {verdict}**", ""]
    if verdict == "IDENTICAL_TO_PREVIOUS":
        L += ["Every file in the new upload is byte-for-byte the same as the one already "
              "analysed. The re-upload contains no new data — only the zip parts were "
              "split differently. The previous conclusion stands: this is not an aperiodic "
              "decomposition, and the non-oscillatory analysis remains blocked on genuinely "
              "new data.", ""]
    elif spec_rows:
        L += ["## Spectral check on the files that changed", "",
              "Alpha prominence is the power in 8–12 Hz relative to a 1/f line fitted on "
              "2–40 Hz excluding 7–14 Hz. A genuine aperiodic-only export should sit at "
              "≈ 0 dB (no oscillatory peak left).", "",
              "| file | new alpha prom. | new 1/f slope | old alpha prom. | old 1/f slope |",
              "|---|---|---|---|---|"]
        for r in spec_rows:
            L.append(f"| `{r['basename']}` | {r['new_alpha_prominence_db']:+.2f} dB | "
                     f"{r['new_1f_slope']:+.3f} | "
                     f"{r.get('old_alpha_prominence_db', float('nan')):+.2f} dB | "
                     f"{r.get('old_1f_slope', float('nan')):+.3f} |")
        L += ["", f"Mean new alpha prominence: "
                  f"**{summary.get('new_alpha_prominence_mean_db')} dB** → "
                  f"{'looks genuinely aperiodic' if summary.get('looks_aperiodic') else 'still contains an oscillatory peak — NOT aperiodic'}",
              ""]
    (out / "new_nonosc_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print("NEW_NONOSC_COMPARE_DONE")


if __name__ == "__main__":
    main()
