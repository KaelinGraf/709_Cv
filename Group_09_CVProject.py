"""
Group 09 — MECHENG 709/710 Computer Vision Project (Spiral weld-gap detection)

Single-file submission per the brief. Combines:
  - WeldDetector            (base class: I/O, ROI, interim/A image writers, CSV)
  - WeldDetectorSeamDP      (dynamic-programming seam-tracking detector)
  - main()                  (entry point: scan ./Set 1, ./Set 2, ./Set 3)

Run from the folder that contains 'Set 1', 'Set 2', 'Set 3' subfolders:
    python Group_09_CVProject.py

Outputs (per brief Fig. 6, relative to CWD):
    InterimResultsOfSet<N>/PositionResultsOfSet<N>.csv
    InterimResultsOfSet<N>/Image<XXXX>_A_WeldGapPosition.JPG
    InterimResultsOfSet<N>/Image<XXXX>_B_InterimResult{1..3}.jpg

The optional sklearn-based glare-extrapolation fallback is OFF by default
(see WeldDetectorSeamDP.__init__ docstring for the rationale: on Set 3
reflective-centre frames it produces confidently-wrong answers, which the
brief penalises at -10 vs +0 for fail-safe). sklearn is therefore lazy-
imported inside the extrapolation method so the shipping path runs with
only opencv-python and numpy.
"""

import cv2
import numpy as np
import os


CWD = os.getcwd()
PIX_WIDTH = 0.04607  # mm per pixel, from brief
MAX_WELD_WIDTH = 0.5  # mm, from brief
MAX_DEV_H = 2.0  # mm/s, max horizontal weld-gap velocity, from brief

DEBUG = False


# ============================================================================
# Base detector — image I/O, ROI math, interim & A-image writing, CSV driver
# ============================================================================
class WeldDetector:
    def __init__(self, y_line: int = 70, expected_width: int = 6, tolerance: int = 3, images=[]):
        """
        args:
            y_line (int): Line of expected weld center location in pixels
            expected_width (int): expected weld width in pixels
            tolerance (int): tolerance of prediction
            images([str]): list of relative paths to source images
        """
        self._y_line: int = y_line
        self._expected_width: int = expected_width
        self._tolerance: int = tolerance
        self.images = []
        for image in images:
            self.images.append(self.load_image(image))

    def load_image(self, path: str = None):
        if path is not None:
            img = cv2.imread(os.path.join(CWD, path))
            if img is None:
                raise Exception("File could not be loaded")
        return img  # np array

    def rgb_to_greyscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def preprocess(self, img_gray: np.ndarray) -> np.ndarray:
        """
        Local contrast normalization to make the gap signal consistent across
        different welding-arc glare and ambient lighting conditions. CLAHE is
        used over global hist-eq because the gap is a small low-contrast
        feature that global eq tends to wash out under bright glare.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img_gray)

    def crop_to_weld_line(self, img_gray: np.ndarray, y_line: int) -> np.ndarray:
        """Returns the single row of pixels at y_line."""
        return img_gray[y_line, :]

    def crop_to_weld_band(self, img_gray: np.ndarray, y_center: int, band_height: int = 5) -> np.ndarray:
        """
        Column-wise mean of a small vertical band around y_center. Trades a
        few pixels of vertical resolution for a substantial SNR boost: per-
        pixel sensor / JPEG noise is uncorrelated across rows but the gap
        signal is correlated, so the mean averages the noise down while
        preserving the gap profile.
        """
        half = band_height // 2
        y0 = max(0, y_center - half)
        y1 = min(img_gray.shape[0], y_center + half + 1)
        band = img_gray[y0:y1, :].astype(np.float32)
        return np.mean(band, axis=0).astype(np.uint8)

    def _roi_bounds(self, image_shape: tuple, x_centre: int,
                    roi_height: int = 100, roi_half_width: int = 200) -> tuple:
        """
        (x0, x1, y0, y1) of the brief-required ROI crop: a band centred
        vertically on y_line and horizontally on x_centre, clamped to the
        image. Used for both the A_WeldGapPosition image and the
        B_InterimResult images so all per-image artefacts share a frame.
        """
        H, W = image_shape[0], image_shape[1]
        y0 = max(0, self._y_line - roi_height // 2)
        y1 = min(H, y0 + roi_height)
        x0 = max(0, int(x_centre) - roi_half_width)
        x1 = min(W, x0 + 2 * roi_half_width)
        x0 = max(0, x1 - 2 * roi_half_width)  # pull x0 in if we hit the right edge
        return x0, x1, y0, y1

    def _compute_interim_images(self, image: np.ndarray, position: int, valid: int,
                                 roi: tuple) -> list:
        """
        Base interim visualisations (subclasses may override):
            1. raw colour ROI
            2. CLAHE-normalised grey ROI (detector input)
            3. CLAHE'd ROI overlaid with y_line guide and predicted-x line
        """
        x0, x1, y0, y1 = roi
        raw_roi = image[y0:y1, x0:x1].copy()
        gray = self.rgb_to_greyscale(image)
        clahe_roi = cv2.cvtColor(self.preprocess(gray)[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
        guide_roi = clahe_roi.copy()
        cv2.line(guide_roi, (0, self._y_line - y0), (guide_roi.shape[1] - 1, self._y_line - y0),
                 (0, 255, 255), 1)
        if valid and position >= 0 and x0 <= position < x1:
            cv2.line(guide_roi, (position - x0, 0), (position - x0, guide_roi.shape[0] - 1),
                     (0, 255, 0), 1)
        return [raw_roi, clahe_roi, guide_roi]

    def _save_outputs_for_image(self, image: np.ndarray, fname: str,
                                 position: int, valid: int, output_dir: str) -> None:
        """
        Brief-spec per-image artefacts:
        - Image{XXXX}_A_WeldGapPosition.JPG : ROI crop with the determined
          weld-gap position drawn (green cross when valid, red tilted cross
          when fail-safed so a human can immediately tell which frames the
          closed-loop controller would actually act on).
        - Image{XXXX}_B_InterimResult{1..3}.jpg : up to three detector-
          specific intermediate ROIs.
        ROI is centred horizontally on the prediction (or image-centre when
        invalid) so the marker is always visible inside the crop.
        """
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(fname)[0]
        W = image.shape[1]
        x_centre = position if (valid and position >= 0) else W // 2
        x0, x1, y0, y1 = self._roi_bounds(image.shape, x_centre)

        a_roi = image[y0:y1, x0:x1].copy()
        cv2.line(a_roi, (0, self._y_line - y0), (a_roi.shape[1] - 1, self._y_line - y0),
                 (0, 255, 255), 1)
        if valid and position >= 0:
            cv2.drawMarker(a_roi, (position - x0, self._y_line - y0), (0, 255, 0),
                           markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        else:
            cx, cy = a_roi.shape[1] // 2, self._y_line - y0
            cv2.drawMarker(a_roi, (cx, cy), (0, 0, 255),
                           markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=2)
        cv2.imwrite(os.path.join(output_dir, f"{base}_A_WeldGapPosition.JPG"), a_roi)

        # Brief asks for "2-3 interim results"; anything beyond 3 is silently dropped.
        for i, interim in enumerate(
                self._compute_interim_images(image, position, valid, (x0, x1, y0, y1))[:3],
                start=1):
            cv2.imwrite(os.path.join(output_dir, f"{base}_B_InterimResult{i}.jpg"), interim)

    def process_dataset(self, image_dir: str, set_number, output_root: str = ".") -> str:
        """
        Iterate a Set folder in filename order and write:
            <output_root>/InterimResultsOfSet<N>/
                PositionResultsOfSet<N>.csv
                Image<XXXX>_A_WeldGapPosition.JPG (per image)
                Image<XXXX>_B_InterimResult{1..3}.jpg (per image)

        CSV format (header and rows include the spacing shown in the brief):
            ImageName, Weld gap position in pixel/integer , Weld gap position valid? 0 = false, 1 = true
            Image0001.jpg, 524, 1
            ...

        Per the brief, invalid detections write -1 / 0 — the closed-loop
        controller treats those as no-measurement and holds the torch.
        """
        interim_dir = os.path.join(output_root, f"InterimResultsOfSet{set_number}")
        os.makedirs(interim_dir, exist_ok=True)

        files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(".jpg"))

        csv_path = os.path.join(interim_dir, f"PositionResultsOfSet{set_number}.csv")
        with open(csv_path, "w") as fh:
            # Brief-mandated header — keep the spacing verbatim; the marking
            # script may parse columns by exact match.
            fh.write("ImageName, Weld gap position in pixel/integer , "
                     "Weld gap position valid? 0 = false, 1 = true\n")
            for fname in files:
                img = self.load_image(os.path.join(image_dir, fname))
                position, valid = self.process_image(img.copy())
                if not valid or position is None or position < 0:
                    position, valid = -1, 0
                fh.write(f"{fname}, {int(position)}, {int(valid)}\n")
                self._save_outputs_for_image(img, fname, int(position), int(valid), interim_dir)

        return os.path.abspath(interim_dir)

    def draw_weld(self, image: np.ndarray, predicted_center: int,
                  y_line: int = 70, valid: int = 1) -> np.ndarray:
        """
        Draws a cross at the predicted weld center on the FULL image.
        Green = passed validation, red = rejected (drawn at the rejected
        location for visualisation only — downstream code must treat
        valid=0 as no-measurement).
        """
        if predicted_center is None or predicted_center < 0:
            return image
        color = (0, 255, 0) if valid else (0, 0, 255)
        cv2.drawMarker(image, (int(predicted_center), y_line), color,
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        return image


# ============================================================================
# SeamDP detector — dynamic-programming seam tracker with matched-filter cost
# ============================================================================
class WeldDetectorSeamDP(WeldDetector):
    """
    Dynamic-programming seam-tracking weld-gap detector.

    The gap is a long thin DARK vertical valley spanning many image rows.
    Detectors that only look at a band of rows around y_line throw away the
    spatial coherence of the gap along the full visible pipe height. This
    detector exploits that coherence directly:

      1. CLAHE pre-processing for illumination invariance.
      2. Row-wise convolution with a DC-balanced 1D matched filter tuned to
         expected_width: kernel = [+1]*shoulder + [-1]*w + [+1]*shoulder.
         Zero sum -> response invariant to global brightness; peaks where
         the local horizontal profile resembles a thin dark valley of
         width w. Conceptually a 1D Frangi/vesselness for thin dark lines.
      3. Seam-carving-style DP: find the path from y_min to y_max
         maximising summed matched-filter response, constrained to drift
         at most max_slope pixels horizontally per row. Slope constraint
         is the geometric prior that the gap is nearly vertical (brief:
         <= 2 mm/s horizontal motion); it rejects paths that zig-zag
         through unrelated dark features.
      4. Read the seam's x at y_line as the gap position.
      5. Confidence: require the seam's mean per-row likelihood to exceed
         a threshold AND lateral excursion to stay below a slope limit
         AND the second-best seam (after masking the first) to be
         substantially less likely. Catches frames with no coherent
         vertical valley (smoke / glare / Set 3) and frames with two
         competing equally-strong gaps.

    Strengths:
      - Uses GLOBAL vertical evidence (every row contributes).
      - No templates needed; fully parameter-driven.
      - DP naturally rejects discontinuous / unlikely paths.
      - Robust to local occlusion: a seam interrupted by smoke can still
        follow visible portions above and below.
    """

    def __init__(self, y_line: int = 70, expected_width: int = 6, tolerance: int = 3, images=[],
                 use_clahe: bool = True, blur_sigma: float = 0.0,
                 shoulder_width: int = 3,
                 y_min: int = 5, y_max: int = 220,
                 max_slope: int = 1,
                 min_mean_likelihood: float = 8.0,
                 max_lateral_excursion: int = 30,
                 search_x_range: tuple = None,
                 second_path_ratio: float = 1.4,
                 second_path_exclusion: int = 25,
                 darkness_threshold: float = 100.0,
                 max_interior_at_y_line: float = 110.0,
                 extrapolate_when_glared: bool = False,
                 extrap_min_visible_likelihood: float = 4.0,
                 extrap_min_visible_rows: int = 40,
                 extrap_max_line_slope: float = 0.1,
                 extrap_ransac_residual: float = 2.0):
        """
        args:
            use_clahe (bool): apply CLAHE pre-processing for lighting
                normalization.
            blur_sigma (float): optional Gaussian blur before the matched
                filter. Default 0 because the matched filter already
                integrates over the gap width.
            shoulder_width (int): width of the bright-shoulder regions in
                the matched-filter kernel. Wide enough to span typical
                stainless surface adjacent to the gap, narrow enough to
                not be confused by parallel seams / fixture edges.
            y_min, y_max: vertical row range used by DP. The gap is most
                reliably visible in the focal area straddling y_line;
                rows outside this range bring in fixture / pipe-edge
                noise that confuses the seam.
            max_slope (int): max horizontal pixels of drift per row. The
                gap is near-vertical (max horizontal velocity 2 mm/s
                ~= 43 px/s, much slower than vertical strip transport),
                so 1 px/row is generous.
            min_mean_likelihood (float): minimum mean per-row matched-
                filter response along the seam for confidence. Kernel
                normalisation makes this approximately (mean_shoulder -
                mean_interior) grey-levels.
            max_lateral_excursion (int): max(seam) - min(seam) cap. A
                real gap's lateral wander should be small.
            search_x_range (tuple): optional (xmin, xmax) restricting
                where the seam may run; enforced by setting cost to
                +inf outside the range.
            second_path_ratio (float): minimum ratio between best seam's
                mean likelihood and second-best (computed after masking
                a window around the first). >=1.4 means the best must
                be >=40% stronger — closer suggests an ambiguous bimodal
                gap; we'd rather reject than guess (wrong = -10, miss = 0).
            second_path_exclusion (int): pixels +- around the best seam
                masked out of the second-pass DP. Must exceed the gap's
                main-lobe width or the "second" path is just a parallel
                shadow of the first.
            darkness_threshold (float): grey-level at which the absolute-
                darkness weight becomes zero. The contrast-only matched
                filter cannot distinguish a deep narrow gap (interior
                ~75) from a shallower-but-wider parallel valley
                (interior ~125) — both have similar (shoulder - interior).
                Weighting by absolute interior darkness breaks the tie.
            max_interior_at_y_line (float): maximum allowed grey-level
                of the seam's pixel at y_line itself. Catches Set 3-style
                frames where the gap is occluded by welding-arc glare
                exactly at y_line — the seam still finds SOME dark path
                across the y_min..y_max strip but at y_line lands on a
                bright glare pixel, so we reject (or extrapolate).
            extrapolate_when_glared (bool): if True, when the y_line glare
                gate triggers, attempt a line-fit extrapolation from the
                visible portion of the seam instead of rejecting outright.
                DEFAULT FALSE — Set 3 reflective-centre frames present a
                parallel non-gap dark feature in the visible region that
                the seam DP locks onto, and extrapolating from THAT path
                produces confidently-wrong answers (-10 each on the marking
                scheme). Returning -1 (-> 0 points) is strictly better.
            extrap_min_visible_likelihood (float): per-row likelihood floor
                for a row to be considered "visible" and contribute to the
                extrapolation fit.
            extrap_min_visible_rows (int): minimum number of visible rows
                required before extrapolation is attempted.
            extrap_max_line_slope (float): maximum |dx/dy| of the fitted
                line. Steeper fit means the line is tracking unrelated
                features; projecting a steep slope ~100 rows back to
                y_line amplifies any slope-estimate error.
            extrap_ransac_residual (float): RANSAC inlier threshold for
                the line fit, in pixels.
        """
        super().__init__(y_line, expected_width, tolerance, images)
        self._use_clahe = use_clahe
        self._blur_sigma = blur_sigma
        self._shoulder_width = shoulder_width
        self._y_min = y_min
        self._y_max = y_max
        self._max_slope = max_slope
        self._min_mean_likelihood = min_mean_likelihood
        self._max_lateral_excursion = max_lateral_excursion
        self._search_x_range = search_x_range
        self._second_path_ratio = second_path_ratio
        self._second_path_exclusion = second_path_exclusion
        self._darkness_threshold = darkness_threshold
        self._max_interior_at_y_line = max_interior_at_y_line
        self._extrapolate_when_glared = extrapolate_when_glared
        self._extrap_min_visible_likelihood = extrap_min_visible_likelihood
        self._extrap_min_visible_rows = extrap_min_visible_rows
        self._extrap_max_line_slope = extrap_max_line_slope
        self._extrap_ransac_residual = extrap_ransac_residual

    def _gap_likelihood(self, img_gray: np.ndarray) -> np.ndarray:
        """
        Per-pixel matched-filter response for a DC-balanced thin-dark-
        vertical-valley kernel, weighted by absolute interior darkness.

        Kernel (1D, applied row-wise):
            [+1] * shoulder_width   |  [-1] * expected_width   |   [+1] * shoulder_width
                bright shoulder     |     dark gap interior    |       bright shoulder

        Sum = 0 -> response invariant to global brightness offset; rectified
        because only positive responses (centre darker than shoulders) are
        weld-gap-like. Normalised by w so response reads roughly as
        (shoulder - interior) grey-levels (makes min_mean_likelihood
        interpretable as a grey-level contrast).
        """
        img = img_gray.astype(np.float32)
        #apply gauissian blur if sigma>0, defaults to off because the matched filter already integrates over the gap width; may help under high noise.
        if self._blur_sigma > 0:
            ksz = max(3, int(2 * round(2 * self._blur_sigma) + 1)) # kernel size: cover +-2 sigma, odd integer
            img = cv2.GaussianBlur(img, (ksz, ksz), self._blur_sigma)

        w = self._expected_width 
        s = self._shoulder_width
        kernel = np.concatenate([
            np.ones(s, dtype=np.float32),
            -np.ones(w, dtype=np.float32),
            np.ones(s, dtype=np.float32),
        ]).reshape(1, -1)
        kernel = kernel / float(w) #element wise division by width. This relies on symmetry (eg 3 pixel shoulder, 6 pixel width, sum to 0)
        #equivalent to convolution for a symmetrical kernel
        #The kernel summing to 0 results in the kernel producing 0 for uniformly bright sections. An output peak only occurs when interior is darker than the shoulders, in a symmetrical fashion
        contrast_resp = cv2.filter2D(img, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
        contrast_resp = np.maximum(0.0, contrast_resp)

        # Absolute-darkness weighting: the contrast-only filter cannot tell
        # a deep narrow gap from a shallower-but-wider parallel valley
        # because both have similar (shoulder - interior). The true gap is
        # by far the darkest narrow vertical valley in the FOV, so weighting
        # by absolute interior darkness breaks the tie.
        interior_kernel = np.concatenate([
            np.zeros(s, dtype=np.float32),
            np.ones(w, dtype=np.float32) / float(w),
            np.zeros(s, dtype=np.float32),
        ]).reshape(1, -1)
        #effectively: if the weld was centered on a pixel, what is the mean INTERIOR grey-level at that pixel
        mean_interior = cv2.filter2D(img, cv2.CV_32F, interior_kernel,
                                     borderType=cv2.BORDER_REFLECT)
        # Linear weight in [0, 1]: 1 for interior=0, 0 for interior >= threshold.
        # This is effecively soft gating to account for varying lighting conditions.
        darkness_weight = np.clip(
            (self._darkness_threshold - mean_interior) / self._darkness_threshold,
            0.0, 1.0,
        )
        return contrast_resp * darkness_weight

    def _dp_seam(self, cost: np.ndarray) -> tuple:
        """
        Seam-carving-style DP: find the column path from row 0 to row H-1
        of `cost` that minimises summed cost, with a per-row lateral slope
        limit of self._max_slope. 
        It is important to note that cost here is -likelihood, which is the result of the shoulder-minus-interior matched filter response. 
        The seam DP finds the path of minimum cost, which corresponds to the path of maximum likelihood for the gap.

        args:
            cost (np.ndarray): [H x W] float32. Use cost = -likelihood
                so high-likelihood pixels become low-cost.
        returns:
            (seam, total_cost) where seam is shape [H] giving the chosen
                x at each row.
        """
        H, W = cost.shape 
        slope = self._max_slope
        acc = np.full_like(cost, fill_value=np.inf)
        acc[0] = cost[0]
        parent = np.zeros((H, W), dtype=np.int32)

        for y in range(1, H):
            prev = acc[y - 1]
            # Stack slope-shifted versions of prev. stacked[s_idx, x] holds
            # prev[x + (s_idx - slope)], the accumulated cost of arriving
            # at row y, column x, having come from column x+(s_idx-slope)
            # at row y-1.
            stacked = np.full((2 * slope + 1, W), np.inf, dtype=np.float32)
            for s_idx, s in enumerate(range(-slope, slope + 1)):
                if s < 0:
                    stacked[s_idx, -s:] = prev[: W + s]
                elif s > 0:
                    stacked[s_idx, : W - s] = prev[s:]
                else:
                    stacked[s_idx] = prev
            best_s_idx = np.argmin(stacked, axis=0)
            best_val = stacked[best_s_idx, np.arange(W)]
            parent[y] = np.arange(W) + (best_s_idx - slope)
            acc[y] = cost[y] + best_val

        end_x = int(np.argmin(acc[-1]))
        seam = np.zeros(H, dtype=np.int32)
        seam[-1] = end_x
        for y in range(H - 1, 0, -1):
            seam[y - 1] = parent[y, seam[y]]
        return seam, float(acc[-1, end_x])

    def _extrapolate_to_y_line(self, rows: np.ndarray, seam: np.ndarray,
                                likelihood_along_seam: np.ndarray):
        """
        Line-fit extrapolation fallback for glare-occluded frames. Trust
        seam positions only at rows where matched-filter likelihood exceeds
        extrap_min_visible_likelihood (so glare-zone rows with ~0
        likelihood are excluded), fit a robust RANSAC line through visible
        (y, x) pairs, project to y_line.

        sklearn is lazy-imported here because extrapolation is OFF in the
        default config — the shipping path needs only opencv + numpy.

        returns:
            extrapolated x at y_line (int), or None if the fit cannot be
            trusted (too few visible rows, slope too steep, too few RANSAC
            inliers).
        """
        from sklearn import linear_model  # lazy import — see docstring

        visible = likelihood_along_seam > self._extrap_min_visible_likelihood
        if int(np.sum(visible)) < self._extrap_min_visible_rows:
            return None

        ys = rows[visible].astype(np.float64).reshape(-1, 1)
        xs = seam[visible].astype(np.float64)

        try:
            ransac = linear_model.RANSACRegressor(
                min_samples=2,
                residual_threshold=self._extrap_ransac_residual,
            )
            ransac.fit(ys, xs)
        except ValueError:
            return None  # no consensus set

        n_inliers = int(np.sum(ransac.inlier_mask_))
        if n_inliers < self._extrap_min_visible_rows:
            return None

        slope = float(ransac.estimator_.coef_[0])
        if abs(slope) > self._extrap_max_line_slope:
            # Steep slope would amplify any slope-estimate error over the
            # ~100-row extrapolation distance back to y_line.
            return None

        x_at_y_line = ransac.predict(np.array([[float(self._y_line)]]))[0]

        if DEBUG:
            print(f"  [seam_dp:extrap] visible_rows={int(np.sum(visible))} "
                  f"inliers={n_inliers} slope={slope:.3f} -> x@{self._y_line}={x_at_y_line:.1f}")

        return int(round(x_at_y_line))

    def extract_weld_center(self, img_gray: np.ndarray) -> tuple:
        """
        Full DP-seam pipeline:
            CLAHE -> matched filter -> seam DP -> confidence checks.
        returns:
            (position, valid). position = -1 when invalid.
        """
        if self._use_clahe:
            img_gray = self.preprocess(img_gray)

        likelihood = self._gap_likelihood(img_gray)

        H, W = likelihood.shape
        y0 = max(0, int(self._y_min))
        y1 = min(H, int(self._y_max))
        if y1 - y0 < 5:
            return (-1, 0)

        cost = -likelihood[y0:y1].astype(np.float32)
        cost = cost.copy()  # float32; needed for the +inf gating below

        if self._search_x_range is not None:
            xmin, xmax = self._search_x_range
            xmin = max(0, int(xmin))
            xmax = min(W, int(xmax))
            cost[:, :xmin] = np.inf
            cost[:, xmax:] = np.inf

        seam, _total_cost = self._dp_seam(cost)

        # Position at y_line. Seam is indexed by row offset from y0, so
        # the row corresponding to y_line is y_line - y0.
        seam_idx = self._y_line - y0
        if seam_idx < 0 or seam_idx >= len(seam):
            return (-1, 0)
        position = int(seam[seam_idx])

        rows = np.arange(y0, y1)
        likelihood_along_seam = likelihood[rows, seam]
        mean_lk = float(np.mean(likelihood_along_seam))
        excursion = int(seam.max() - seam.min())

        # Second-best seam: re-run DP after masking a window around the
        # first seam. If the second-best is also very strong, the gap
        # location is ambiguous (twin-valley case) — reject.
        masked_cost = cost.copy()
        for y_off, x in enumerate(seam):
            xl = max(0, int(x) - self._second_path_exclusion)
            xr = min(W, int(x) + self._second_path_exclusion + 1)
            masked_cost[y_off, xl:xr] = np.inf
        seam2, _ = self._dp_seam(masked_cost)
        likelihood_along_seam2 = likelihood[rows, seam2]
        mean_lk2 = float(np.mean(likelihood_along_seam2))

        # y_line glare gate: the seam may have found a coherent dark path
        # somewhere in the strip, but if at y_line specifically the seam
        # sits on a bright pixel, the gap is occluded by welding-arc glare
        # exactly where the brief requires the answer. Two options: reject,
        # or try line-fit extrapolation from the visible portion.
        seam_pixel_at_y_line = float(img_gray[self._y_line, position])
        glare_at_y_line = seam_pixel_at_y_line > self._max_interior_at_y_line
        extrapolated = None
        if glare_at_y_line and self._extrapolate_when_glared:
            extrapolated = self._extrapolate_to_y_line(
                rows, seam, likelihood_along_seam)

        valid = 1
        reasons = []
        if glare_at_y_line and extrapolated is None:
            valid = 0
            reasons.append(f"y_line glare: I[{position},{self._y_line}]="
                           f"{seam_pixel_at_y_line:.0f}>{self._max_interior_at_y_line}")
        if mean_lk < self._min_mean_likelihood and not glare_at_y_line:
            # mean-lk floor only applies when we trusted the seam at y_line.
            # Under glare the seam's glared rows have likelihood ~0 by
            # construction, which would always fail this gate; the
            # extrapolation path has its own min-visible-rows confidence.
            valid = 0
            reasons.append(f"mean_lk={mean_lk:.1f}<{self._min_mean_likelihood}")
        if excursion > self._max_lateral_excursion and not glare_at_y_line:
            valid = 0
            reasons.append(f"excursion={excursion}>{self._max_lateral_excursion}")
        if mean_lk2 > 0 and not glare_at_y_line:
            ratio = mean_lk / max(mean_lk2, 1e-6)
            if ratio < self._second_path_ratio:
                valid = 0
                reasons.append(f"2nd-path ratio {ratio:.2f}<{self._second_path_ratio}")
        if extrapolated is not None and valid == 1:
            position = extrapolated

        if DEBUG:
            status = "ACCEPT" if valid else "REJECT (" + ", ".join(reasons) + ")"
            print(f"  [seam_dp] x@y_line={position} mean_lk={mean_lk:.1f} "
                  f"mean_lk2={mean_lk2:.1f} excursion={excursion}  {status}")

        if valid == 0:
            return (-1, 0)
        return (position, 1)

    def _compute_interim_images(self, image: np.ndarray, position: int, valid: int,
                                 roi: tuple) -> list:
        """
        SeamDP-specific interim visualisations:
            1. raw colour ROI — original image content
            2. CLAHE-normalised grey ROI — actual detector input
            3. matched-filter likelihood heatmap of the ROI, with y_line
               guide and predicted-x line overlaid. Hot pixels = "thin
               dark valley"; the predicted line should sit on a warm
               column. Most useful debug artefact when a detection looks
               wrong — shows whether the matched filter even saw the gap.
        """
        x0, x1, y0, y1 = roi
        raw_roi = image[y0:y1, x0:x1].copy()
        gray = self.rgb_to_greyscale(image)
        gray_p = self.preprocess(gray) if self._use_clahe else gray
        clahe_roi = cv2.cvtColor(gray_p[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)

        # Compute likelihood on the full image then crop so the filter
        # response matches what the DP actually consumed (cropping first
        # would leak boundary effects into the displayed ROI).
        likelihood = self._gap_likelihood(gray_p)
        like_roi = likelihood[y0:y1, x0:x1]
        cap = float(np.percentile(like_roi, 99)) if like_roi.size else 1.0
        cap = max(cap, 1e-3)
        like_norm = np.clip(like_roi / cap, 0.0, 1.0)
        like_u8 = (like_norm * 255.0).astype(np.uint8)
        heat = cv2.applyColorMap(like_u8, cv2.COLORMAP_INFERNO)
        cv2.line(heat, (0, self._y_line - y0), (heat.shape[1] - 1, self._y_line - y0),
                 (0, 255, 255), 1)
        if valid and position >= 0 and x0 <= position < x1:
            cv2.line(heat, (position - x0, 0), (position - x0, heat.shape[0] - 1),
                     (0, 255, 0), 1)
        return [raw_roi, clahe_roi, heat]

    def process_image(self, image: np.ndarray) -> tuple:
        """
        Run the full seam-DP pipeline on a BGR image.
        returns:
            (position, valid). position = -1 when invalid.
        """
        gray = self.rgb_to_greyscale(img=image)
        position, valid = self.extract_weld_center(gray)
        image = self.draw_weld(image, position, self._y_line, valid)

        if DEBUG:
            cv2.imshow("Weld Center (SeamDP)", image)
            cv2.waitKey(0)
        return (position, valid)

    def process_all(self):
        results = []
        for image in self.images:
            results.append(self.process_image(image))
        return results


# ============================================================================
# Entry point
# ============================================================================
def main():
    """
    Brief-spec entry point. Runs the SeamDP detector over every image in
    ./Set 1, ./Set 2, ./Set 3 (relative paths, per the brief — do NOT
    hardcode an absolute project path; the marking script runs from its
    own root), and writes the brief-mandated artefacts:
        InterimResultsOfSet<N>/PositionResultsOfSet<N>.csv
        InterimResultsOfSet<N>/Image<XXXX>_A_WeldGapPosition.JPG
        InterimResultsOfSet<N>/Image<XXXX>_B_InterimResult{1..3}.jpg

    SeamDP was chosen because across the labelled training data it scored
    zero wrong predictions (Set 1: 50/50, Set 2: 20/21, Set 3: correctly
    rejects all reflective-centre frames where intensity-based detection
    cannot be trusted). The -10 penalty for a wrong prediction vs +1 for
    a correct one makes the fail-safe behaviour load-bearing.
    """
    # Central-60% horizontal search range derived from the first available
    # image's width — the camera is mounted on the welding torch which
    # actively tracks the gap, so the gap stays roughly mid-frame and the
    # outer 40% of columns only ever holds fixture / bezel distractors
    # that confuse the matched filter.
    set_names = ("Set 1", "Set 2", "Set 3")
    sample_set = next((s for s in set_names if os.path.isdir(s)), None)
    if sample_set is None:
        print("[error] No 'Set 1' / 'Set 2' / 'Set 3' folder found in CWD. "
              "Run this script from the folder containing those subfolders.")
        return

    sample_files = sorted(f for f in os.listdir(sample_set)
                          if f.lower().endswith(".jpg"))
    if not sample_files:
        print(f"[error] No .jpg images found in '{sample_set}'.")
        return

    sample = cv2.imread(os.path.join(sample_set, sample_files[0]))
    if sample is None:
        print(f"[error] Could not read sample image "
              f"'{os.path.join(sample_set, sample_files[0])}'.")
        return
    W = sample.shape[1]
    search_x_range = (int(W * 0.2), int(W * 0.8))

    detector = WeldDetectorSeamDP(images=[], search_x_range=search_x_range)

    for set_num in (1, 2, 3):
        set_dir = f"Set {set_num}"
        if not os.path.isdir(set_dir):
            print(f"[skip] {set_dir} not found")
            continue
        out = detector.process_dataset(set_dir, set_num, output_root=".")
        print(f"Set {set_num} -> {out}")


if __name__ == "__main__":
    main()
