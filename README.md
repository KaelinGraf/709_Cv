# Weld-Gap Detection (706 Project 1)

Vision-based weld-gap position detection for closed-loop welding-torch
control. Given a JPG frame from the torch-mounted camera, find the
horizontal pixel position of the weld gap at a fixed measurement row
(`y_line = 70`) and report whether the measurement is trustworthy.

The shipping detector is [**SeamDP**](detectors/seam_dp.py), a dynamic-
programming seam-tracking detector that exploits the gap's vertical
coherence across the whole frame. See
[seam_dp_explain.md](seam_dp_explain.md) for a step-by-step walkthrough,
including the role of CLAHE preprocessing.

---

## Quick start

```bash
# Run the chosen (SeamDP) detector over Set 1, Set 2, Set 3 and write the
# brief-mandated outputs into ./InterimResultsOfSet{1,2,3}/
python weld_detect.py

# Score every detector against the labelled ground truth
python -m utils.evaluate

# Score one detector with per-image detail
python -m utils.evaluate --detector seam_dp --verbose
```

Inputs are read from `WeldGapImages_export/Set {1,2,3}/Image*.jpg`
using **relative paths** (per the brief — the marking script runs from
its own working directory, so absolute paths would break it).

---

## Marking scheme

The detector is scored frame-by-frame:

| Outcome                                              | Set 1 / 2 | Set 3 |
| ---------------------------------------------------- | --------- | ----- |
| Correct (`valid=1`, `|pos − truth| ≤ 3 px`)          | +1        | +7    |
| Wrong (`valid=1` but outside tolerance, or no gap)   | −10       | −10   |
| Miss (`valid=0`)                                     | 0         | 0     |

Wrong predictions cost **10× more** than misses earn, so the right
strategy is to refuse to guess on ambiguous frames. Every confidence
gate in [`seam_dp.py`](detectors/seam_dp.py) is calibrated against this
asymmetry.

---

## Output format (per the brief)

For each set, [`process_dataset`](detectors/base.py#L205) writes:

```
InterimResultsOfSet<N>/
    PositionResultsOfSet<N>.csv
    Image<XXXX>_A_WeldGapPosition.JPG       (final detection cross)
    Image<XXXX>_B_InterimResult{1..3}.jpg   (per-detector debug views)
```

CSV format (header spacing is verbatim from the brief and must not be
reflowed — the marking script may parse columns by exact match):

```
ImageName, Weld gap position in pixel/integer , Weld gap position valid? 0 = false, 1 = true
Image0001.jpg, 524, 1
Image0002.jpg, -1, 0
...
```

Invalid detections are normalised to `position = -1, valid = 0` so the
closed-loop controller treats them as no-measurement and holds the
torch.

---

## Project layout

```
weld_detect.py              Entry point — runs SeamDP over all 3 sets.
detectors/
    base.py                 WeldDetector base class: I/O, ROI crop,
                            CLAHE preprocessing, output writers.
    seam_dp.py              SeamDP detector (the one we ship).
    template_matching.py    ZNCC-template / RANSAC detector (baseline).
    canny.py                Canny-edge detector (baseline).
utils/
    evaluate.py             Score detectors against ground_truth/.
    label_ground_truth.py   Manual labelling tool (writes set_*.json).
    template_extractor.py   Sample template patches from labelled gaps.
    resize_templates.py     Build the templates_25/ multi-scale set.
ground_truth/
    set_{1,2,3}.json        {filename: x-pixel}, x = -1 means no gap.
templates/, templates_25/   Template patches for the TM detector.
WeldGapImages_export/
    Set 1/, Set 2/, Set 3/  Source frames (gitignored).
seam_dp_explain.md          Deep-dive explanation of the SeamDP pipeline.
```

---

## The three detectors

All three share the [`WeldDetector`](detectors/base.py#L14) base class
(image I/O, CLAHE preprocessing, ROI cropping, brief-spec output
writing). They differ only in `extract_weld_center` / `process_image`.

### SeamDP — `WeldDetectorSeamDP` *(shipping)*

Dynamic-programming seam tracker:

1. CLAHE for illumination invariance.
2. DC-balanced 1D matched filter tuned to `expected_width`, weighted by
   absolute interior darkness.
3. Seam-carving DP across all rows in `y_min..y_max`, slope-limited to
   ±1 px/row (gap is near-vertical).
4. Confidence gates: mean per-row likelihood, lateral excursion,
   second-best-seam ratio, and a glare-at-`y_line` check.

Uses **global vertical evidence** (every row contributes), where
template-matching and Canny only sample a narrow band around `y_line`.
This makes it robust to local occlusion and removes the need for
hand-curated templates. Full walkthrough in
[seam_dp_explain.md](seam_dp_explain.md).

### Template matching — `WeldDetectorTemplateMatching`

ZNCC mixture-of-experts across many templates sampled from labelled
gaps, with multi-line RANSAC at evaluation rows around `y_line` to
reject outlier rows. Confidence is gated on PSR and template-vote
agreement.

### Canny edges — `WeldDetectorCanny`

Edge-based: pair the two nearest near-vertical edges with the right
spacing for a gap, then fit a centre.

The two baselines are kept in the repo for comparison and ablation; the
shipping path is SeamDP.

---

## Configuration knobs that matter

The most tunable parameters live on the SeamDP constructor in
[`seam_dp.py`](detectors/seam_dp.py#L49):

- `expected_width` (px) — the gap width the matched filter is tuned for.
- `min_mean_likelihood` — minimum average shoulder-vs-interior
  contrast (grey levels) along the seam to consider it a real gap.
- `max_lateral_excursion` (px) — how far the seam may wander
  laterally; rejects zig-zag paths through unrelated dark features.
- `second_path_ratio` — best/second-best mean-likelihood ratio
  required; rejects ambiguous twin-valley frames.
- `max_interior_at_y_line` (grey level) — rejects frames where the
  seam lands on a glare pixel exactly at the measurement row.
- `search_x_range` — column range the seam may run in. `weld_detect.py`
  sets this to the central 60% of the frame because the torch-mounted
  camera tracks the gap, so it never strays into the outer columns.
- `extrapolate_when_glared` — line-fit fallback when `y_line` is
  glared. **Off by default** — Set 3 reflective-centre frames produce
  confidently-wrong extrapolations and −10 each is much worse than
  returning −1 for 0.

---

## Evaluation

[`utils/evaluate.py`](utils/evaluate.py) loads per-set ground truth
(`{filename: x_truth}`, `x_truth = -1` means "no visible gap") and
prints a per-detector × per-set score table:

```
detector   set     correct wrong  miss noG✓ noG✗  score   /of
----------------------------------------------------------------------
SeamDP     Set 1        50     0     0    0    0    +50    50
SeamDP     Set 2        20     0     1    0    0    +20    21
SeamDP     Set 3         0     0     0   16    0      0    16
```

Followed by a list of every wrong / mismatched prediction with
predicted-vs-truth coordinates, so regressions can be diagnosed
quickly.
