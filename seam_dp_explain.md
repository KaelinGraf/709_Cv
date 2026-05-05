# Seam-DP Weld-Gap Extraction — Step-by-Step Explanation

This document walks through how [`WeldDetectorSeamDP`](detectors/seam_dp.py)
finds the weld gap in an image, from the raw BGR frame to a final
`(position, valid)` answer. The entry point is
[`process_image`](detectors/seam_dp.py#L509) which calls
[`extract_weld_center`](detectors/seam_dp.py#L362).

---

## What is CLAHE? (the preprocessing step)

**CLAHE** = **C**ontrast **L**imited **A**daptive **H**istogram
**E**qualisation.

Used in [`base.py:preprocess`](detectors/base.py#L59-L71):

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
return clahe.apply(img_gray)
```

### Plain-English version
A normal "histogram equalisation" stretches the brightness range of an
image so dark pixels get darker and bright pixels get brighter — this
boosts contrast. The problem: if you have a single very bright welding
arc in one corner of the frame, *global* histogram equalisation tries to
accommodate that one extreme, and the rest of the image gets washed out.

CLAHE fixes both problems:

1. **Adaptive (the "A" in CLAHE)** — it splits the image into a grid
   (here `8 x 8` tiles) and equalises each tile **locally**. So a tile
   sitting under the welding arc gets equalised independently of a tile
   on the dark side of the pipe. Each region of the image ends up with
   good local contrast regardless of what's happening elsewhere.

2. **Contrast Limited (the "CL")** — without a limit, equalisation can
   amplify noise hugely in nearly-uniform regions (because it tries to
   spread small intensity differences across the full 0–255 range). The
   `clipLimit=2.0` parameter caps how aggressively any single grey level
   can be amplified, so smooth regions stay smooth instead of becoming
   speckly.

After CLAHE, tile boundaries are smoothed by bilinear interpolation so
you don't see grid artefacts.

### Why it matters for *this* detector
The weld gap is a **thin, low-contrast dark feature** that lives next to
a **very bright** welding arc. With raw pixels, the gap's local contrast
varies wildly between frames depending on glare. After CLAHE the gap-vs-
shoulder contrast is roughly the same in every frame, which means the
matched-filter and likelihood thresholds downstream can use one fixed
configuration across the whole dataset.

The block comment in `preprocess` says exactly this:
> CLAHE is used over global hist-eq because the gap is a small
> low-contrast feature that global eq tends to wash out under bright
> glare.

---

## The Pipeline at a Glance

```
BGR image
    |
    v
[1] BGR -> greyscale            (rgb_to_greyscale)
    |
    v
[2] CLAHE preprocessing         (preprocess)
    |
    v
[3] Matched-filter likelihood   (_gap_likelihood)
    |
    v
[4] Seam-DP path search         (_dp_seam)
    |
    v
[5] Read x at y_line            (seam[y_line - y0])
    |
    v
[6] Confidence gates            (mean_lk, excursion, 2nd path, glare)
    |
    v
[7] Optional line-fit fallback  (_extrapolate_to_y_line)
    |
    v
(position, valid)
```

---

## Step 1 — Convert to Greyscale

[`rgb_to_greyscale`](detectors/base.py#L56-L57) calls
`cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`.

The gap is a **luminance** feature — there is no useful colour
information about whether a pixel is "inside the gap" or "next to it".
Working in a single channel makes everything that follows simpler and
~3× cheaper, and avoids weighting decisions across R/G/B.

---

## Step 2 — CLAHE Preprocessing

[`preprocess`](detectors/base.py#L59-L71) — see the long explanation
above. After this step, the gap and its bright shoulders have a
consistent contrast across all frames in the dataset, regardless of how
much arc glare or ambient light is in any individual frame.

---

## Step 3 — Matched-Filter Likelihood

[`_gap_likelihood`](detectors/seam_dp.py#L188-L256).

The goal of this step is to convert the greyscale image into a
**per-pixel score** that says "how much does the local horizontal
intensity profile around this pixel look like a thin dark vertical
valley?". The output is the same shape as the input image, in float32.

It does this in two parts:

### 3a. Contrast response (DC-balanced matched filter)

A 1D kernel is convolved across each row:

```
[+1] * shoulder_width   |   [-1] * expected_width   |   [+1] * shoulder_width
   bright shoulder      |       dark gap interior   |       bright shoulder
```

Properties:

- **Sums to zero** — so a uniformly bright OR uniformly dark patch
  responds with `0`. Only **local contrast** matters, not absolute
  brightness. This makes the response invariant to global illumination
  changes between frames.
- **Normalised by `expected_width`** — the response numerically
  approximates `(mean_shoulder_intensity - mean_interior_intensity)` in
  grey levels, so the threshold `min_mean_likelihood = 8.0` is
  interpretable as "the gap must be at least 8 grey levels darker than
  its shoulders, on average, along its length".
- **Rectified** (`np.maximum(0, …)`) — negative responses correspond to
  *bright* bumps on dark backgrounds (e.g. a thin reflection); not weld
  gaps, so we discard them.

This produces `contrast_resp` with high values where the local
horizontal cross-section looks like `bright | dark | bright` of the
expected width.

### 3b. Absolute-darkness weighting

The contrast filter alone has a known weakness: it can't distinguish a
**deep narrow gap** (interior ~75) from a **shallower wider valley**
(interior ~125), because both have similar `(shoulder − interior)`
contrast.

So we multiply the contrast response by a `darkness_weight` that goes
from `1.0` (interior pixel is pitch black) down to `0.0` (interior pixel
is at or above `darkness_threshold = 100`). This is computed by
convolving with a "mean-of-interior" kernel and applying a clipped
linear ramp.

The final likelihood is:

```
likelihood = contrast_resp * darkness_weight
```

i.e. *"narrow valley AND truly dark interior"*.

---

## Step 4 — Seam-Carving-Style Dynamic Programming

[`_dp_seam`](detectors/seam_dp.py#L258-L305).

The likelihood image tells us per pixel "does this look like a gap?",
but a gap is not just one pixel — it's a **near-vertical line** of high-
likelihood pixels spanning many rows. We need to find the best such
line.

This is the same algorithm behind seam-carving for image resizing:

1. **Cost = −likelihood** — we want to *minimise* cost, so high-
   likelihood pixels become low-cost pixels.
2. **Restrict to the strip `y0..y1`** = `(y_min, y_max)` = `(5, 220)`.
   Rows outside this band are dominated by fixture / pipe-edge noise.
3. **Optional `search_x_range`** — set cost to `+inf` outside that
   range so the seam can't enter those columns.
4. **Forward pass** — fill an accumulator `acc[y, x]` = "minimum total
   cost of any path from row 0 to `(y, x)` that respects the slope
   constraint". The slope constraint is `max_slope = 1`, meaning the
   path can only step ±1 column per row. This is the geometric prior
   that the gap is nearly vertical.
5. **Vectorisation trick** — instead of looping over (`y`, `x`, `Δx`),
   we stack `(2 * slope + 1)` shifted copies of the previous row and
   take the per-column min. The Python loop is just over `H` rows;
   the inner work is numpy.
6. **Backward pass** — pick the column at the bottom row with the
   smallest accumulated cost, then walk `parent[y, x]` upwards to
   reconstruct the seam. Output: `seam[y]` = the chosen `x` for each
   row in the strip.

The result is a 1D array of x-coordinates, one per row in `y0..y1`,
tracing the **single best near-vertical dark line** in the image.

---

## Step 5 — Read the Position at `y_line`

The brief asks for the gap's `x` at one specific row, `y_line = 70`.
The seam is indexed from `y0`, so the correct array index is
`y_line - y0`. That x-value is the candidate `position`.

---

## Step 6 — Confidence Gates

A seam will *always* be found, even on a frame full of smoke and glare
that has no real gap visible. We need to decide whether to **trust**
this seam. The detector applies four gates:

### 6a. Mean per-row likelihood

```
mean_lk = mean(likelihood along the seam)
require mean_lk >= min_mean_likelihood   (default 8.0)
```

Because the kernel is normalised, this reads as "the gap must be on
average ≥8 grey levels darker than its shoulders along its length".
A clean weld typically sits around 25-80; below ~8 we're tracking
texture noise.

### 6b. Lateral excursion

```
excursion = max(seam) - min(seam)
require excursion <= max_lateral_excursion   (default 30 px)
```

A real gap doesn't wander 100 pixels left-right. If the seam does, it
is stitching together unrelated dark features under the slope
constraint.

### 6c. Second-best-seam ratio

The DP is re-run on a **masked** cost map where a window of
`±second_path_exclusion = 25` columns around the first seam is set to
`+inf`. This forces the second pass to find the next-best path that
isn't a parallel shadow of the first.

```
ratio = mean_lk / mean_lk2
require ratio >= second_path_ratio   (default 1.4)
```

If two equally-strong gap candidates exist (twin-valley case), the
detector returns invalid rather than guess. The marking scheme is
**−10 points for a wrong answer vs +1 for a hit**, so refusing to guess
when ambiguous is the rational play.

### 6d. Glare-at-`y_line` gate

```
seam_pixel_at_y_line = img_gray[y_line, position]
glare = seam_pixel_at_y_line > max_interior_at_y_line   (default 110)
```

Catches Set 3 frames where the gap is occluded by welding-arc glare
**exactly at `y_line`**. The DP can still find a coherent dark seam
through the rest of the strip, but at `y_line` itself the seam lands on
a bright glare pixel, so the answer at the brief's measurement row is
unreliable. Reject — unless the line-fit fallback below kicks in.

---

## Step 7 — Optional Line-Fit Extrapolation

[`_extrapolate_to_y_line`](detectors/seam_dp.py#L307-L360). Only enabled
when `extrapolate_when_glared=True` (off by default).

When the seam is glared at `y_line` but visible elsewhere, we can
sometimes recover by:

1. Selecting the rows along the seam with `likelihood >
   extrap_min_visible_likelihood` — i.e. rows where the seam is
   actually tracking the gap, not glare zones with `~0` likelihood.
2. Requiring at least `extrap_min_visible_rows` such rows.
3. Fitting a **RANSAC line** through those `(y, x)` points, with a
   `extrap_ransac_residual = 2 px` inlier threshold to reject points
   where the seam wandered.
4. Rejecting the fit if `|slope| > extrap_max_line_slope = 0.1`,
   because a steep slope amplifies any error in slope estimation over
   the ~100-row extrapolation distance.
5. Predicting `x` at `y_line` from the fitted line.

**Why is this OFF by default?** Set 3 has reflective weld-centre frames
where a *parallel non-gap dark feature* exists in the visible region.
The seam DP locks onto that, and extrapolating from it produces
confidently-wrong answers — −10 each on the marking scheme. Returning
−1 (0 points) is strictly better than guessing wrong.

---

## Step 8 — Final Output

```
if valid == 0:    return (-1, 0)
else:             return (position, 1)
```

Then [`process_image`](detectors/seam_dp.py#L509-L523) draws a
green/red cross at the predicted location for visualisation and
returns the same tuple.

The downstream
[`process_dataset`](detectors/base.py#L205-L249) writes:

- `Image{XXXX}_A_WeldGapPosition.JPG` — final detection cross
- `Image{XXXX}_B_InterimResult{1..3}.jpg` — for SeamDP these are
  raw ROI / CLAHE ROI / matched-filter heatmap (see
  [`_compute_interim_images`](detectors/seam_dp.py#L466-L507))
- `PositionResultsOfSet{N}.csv` — per-image (filename, x, valid)

---

## Why this design?

The two weaknesses of a naïve "look at one row of pixels" detector
(which is what the template-matching and Canny detectors essentially
do) are:

1. They use only the rows around `y_line`, throwing away the gap's
   coherence along its full length.
2. They have nothing to fall back on when those specific rows are
   occluded.

Seam-DP exploits the fact that **the gap is a long thin vertical
feature** to integrate evidence across the whole image height, then
uses confidence gates that are stated in physically-meaningful units
(grey levels of contrast, pixels of lateral wander) so they generalise
across sets without per-set tuning.
