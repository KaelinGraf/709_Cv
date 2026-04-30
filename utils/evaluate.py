"""
Evaluate a detector against ground-truth labels.

Run from project root:
    python -m utils.evaluate

Reads labels from ground_truth/set_<n>.json (-1 means "no visible gap").
Scores each detector by the project's marking scheme:
    +1 (Set 1/2) or +7 (Set 3) per correct prediction
    -10 per wrong prediction (predicted valid, but |pos - truth| > 3,
        OR predicted valid but truth is -1)
    0 for misses (predicted invalid)
Also reports the per-frame outcome summary so we can see exactly which
images each detector gets right / wrong / misses.
"""
import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from detectors import (
    WeldDetectorTemplateMatching,
    WeldDetectorCanny,
    WeldDetectorSeamDP,
)


TOLERANCE = 3            #brief-defined +- pixel tolerance
SET_REWARD = {"Set 1": 1, "Set 2": 1, "Set 3": 7}
WRONG_PENALTY = -10


def evaluate_detector(name, detector, sets, labels_by_set, verbose=False):
    """Returns dict of per-set stats and a flat list of per-image rows."""
    rows = []
    summary = {}
    for set_name, files in sets.items():
        labels = labels_by_set.get(set_name, {})
        n_correct = 0
        n_wrong = 0
        n_miss = 0
        n_no_gap_correct = 0    #truth=-1, predicted invalid (correct)
        n_no_gap_wrong = 0      #truth=-1, predicted valid (false positive)
        score = 0
        for path in files:
            fname = os.path.basename(path)
            if fname not in labels:
                continue
            truth = labels[fname]
            img = cv2.imread(path)
            pos, valid = detector.process_image(img.copy())

            outcome = ""
            if truth == -1:
                if valid == 0:
                    outcome = "correct (no-gap)"
                    n_no_gap_correct += 1
                else:
                    outcome = f"WRONG (no-gap; predicted {pos})"
                    n_no_gap_wrong += 1
                    score += WRONG_PENALTY
            else:
                if valid == 1:
                    if abs(pos - truth) <= TOLERANCE:
                        outcome = f"correct ({pos} vs {truth})"
                        n_correct += 1
                        score += SET_REWARD[set_name]
                    else:
                        outcome = f"WRONG ({pos} vs {truth})"
                        n_wrong += 1
                        score += WRONG_PENALTY
                else:
                    outcome = f"miss (truth={truth})"
                    n_miss += 1

            rows.append({"detector": name, "set": set_name, "image": fname,
                         "truth": truth, "pos": pos, "valid": valid, "outcome": outcome})
            if verbose:
                print(f"  [{set_name}] {fname}: {outcome}")

        summary[set_name] = {
            "correct": n_correct,
            "wrong": n_wrong,
            "miss": n_miss,
            "no_gap_correct": n_no_gap_correct,
            "no_gap_wrong": n_no_gap_wrong,
            "score": score,
            "n_labeled": len(labels),
        }
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="WeldGapImages_export")
    ap.add_argument("--labels_dir", default="ground_truth")
    ap.add_argument("--detector", default="all",
                    choices=["all", "tm", "canny", "seam_dp"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    set_names = ["Set 1", "Set 2", "Set 3"]
    sets = {s: sorted(os.path.join(args.root, s, f)
                       for f in os.listdir(os.path.join(args.root, s))
                       if f.lower().endswith(".jpg"))
            for s in set_names if os.path.isdir(os.path.join(args.root, s))}

    labels_by_set = {}
    for s in set_names:
        safe = s.replace(" ", "_").lower()
        path = os.path.join(args.labels_dir, f"{safe}.json")
        if os.path.exists(path):
            with open(path) as fh:
                labels_by_set[s] = json.load(fh)

    sample = cv2.imread(sets[set_names[0]][0])
    W = sample.shape[1]
    sxr = (int(W * 0.2), int(W * 0.8))

    detectors = {}
    if args.detector in ("all", "tm"):
        detectors["TM"] = WeldDetectorTemplateMatching(images=[], search_x_range=sxr)
    if args.detector in ("all", "canny"):
        detectors["Canny"] = WeldDetectorCanny(images=[], search_x_range=sxr)
    if args.detector in ("all", "seam_dp"):
        detectors["SeamDP"] = WeldDetectorSeamDP(images=[], search_x_range=sxr)

    all_rows = []
    print(f"\n{'detector':<10} {'set':<7} {'correct':>7} {'wrong':>5} {'miss':>5} "
          f"{'noG✓':>5} {'noG✗':>5} {'score':>6} {'/of':>5}")
    print("-" * 70)
    for name, det in detectors.items():
        summary, rows = evaluate_detector(name, det, sets, labels_by_set,
                                           verbose=args.verbose)
        all_rows.extend(rows)
        for s in set_names:
            if s not in summary:
                continue
            stats = summary[s]
            print(f"{name:<10} {s:<7} {stats['correct']:>7} {stats['wrong']:>5} "
                  f"{stats['miss']:>5} {stats['no_gap_correct']:>5} {stats['no_gap_wrong']:>5} "
                  f"{stats['score']:>+6} {stats['n_labeled']:>5}")

    #per-detector failure detail
    if not args.verbose:
        print("\n--- failures (predicted-vs-truth mismatches) ---")
        for r in all_rows:
            if "WRONG" in r["outcome"]:
                print(f"  [{r['detector']:<8}] [{r['set']}] {r['image']}: {r['outcome']}")


if __name__ == "__main__":
    main()
