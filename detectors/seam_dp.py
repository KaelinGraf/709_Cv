import cv2
import numpy as np
from sklearn import linear_model

from detectors.base import WeldDetector, DEBUG


class WeldDetectorSeamDP(WeldDetector):
    """
    Dynamic-programming seam-tracking weld-gap detector.

    The gap is a long thin DARK vertical valley spanning many image rows.
    Both the template and Canny detectors only look at a narrow band of
    rows around y_line, throwing away the spatial coherence of the gap
    along the full visible height of the pipe. This detector exploits
    that coherence directly:

    1. Pre-process the image with CLAHE for illumination invariance.
    2. Convolve every row with a DC-balanced 1D matched filter tuned to
       expected_width: kernel = [+1]*shoulder + [-1]*w + [+1]*shoulder.
       The kernel sum is zero so the response is invariant to global
       brightness shifts; the response peaks where the local horizontal
       intensity profile resembles a thin dark valley of width w.
       (Conceptually equivalent to a 1D Frangi/vesselness response for
       the case of thin dark linear features.)
    3. Use seam-carving-style dynamic programming to find the path from
       y_min to y_max that maximises the summed matched-filter response,
       constrained to drift at most max_slope pixels horizontally per
       row. Constraining slope is the geometric prior that the gap is
       nearly vertical (the brief specifies <= 2 mm/s horizontal motion);
       it rejects paths that zig-zag through unrelated dark features.
    4. Read the seam's x at y_line as the gap position.
    5. Confidence: require the seam's mean per-row likelihood to exceed
       a threshold AND the seam's lateral excursion to stay below a slope
       limit AND the SECOND-best seam (after masking out the first) to
       be substantially less likely. Catches frames where there is no
       coherent vertical valley (smoke / heavy glare / Set 3 quality)
       and frames with two competing equally-strong gaps.

    Strengths over template / Canny:
    - Uses GLOBAL vertical evidence (every row contributes), not just
      one band of rows around y_line.
    - No templates needed — fully parameter-driven.
    - DP naturally rejects discontinuous / unlikely paths.
    - Robust to local occlusion: if part of the gap is hidden behind
      smoke or splatter, the seam follows the visible portions on
      either side as long as enough remain to satisfy the confidence.
    """
    def __init__(self, y_line:int = 70, expected_width:int = 6, tolerance:int = 3, images=[],
                 use_clahe:bool = True, blur_sigma:float = 0.0,
                 shoulder_width:int = 3,
                 y_min:int = 5, y_max:int = 220,
                 max_slope:int = 1,
                 min_mean_likelihood:float = 8.0,
                 max_lateral_excursion:int = 30,
                 search_x_range:tuple = None,
                 second_path_ratio:float = 1.4,
                 second_path_exclusion:int = 25,
                 darkness_threshold:float = 100.0,
                 max_interior_at_y_line:float = 110.0,
                 extrapolate_when_glared:bool = False,
                 extrap_min_visible_likelihood:float = 4.0,
                 extrap_min_visible_rows:int = 40,
                 extrap_max_line_slope:float = 0.1,
                 extrap_ransac_residual:float = 2.0):
        """
        args:
            use_clahe (bool): apply CLAHE pre-processing for lighting
                normalization.
            blur_sigma (float): optional small Gaussian blur before the
                matched filter. Default 0 because the matched filter
                already integrates over the gap width and adding a blur
                step risks smoothing the narrow gap into its surroundings.
            shoulder_width (int): width of the bright-shoulder region on
                each side of the dark centre in the matched-filter
                kernel. Should be a few pixels — wide enough to span
                the typical bright stainless surface adjacent to the
                gap, narrow enough not to be confused by adjacent
                features (parallel seams, fixture edges).
            y_min, y_max: vertical range of rows used by DP. The gap is
                most reliably visible in the camera's focal area
                straddling y_line; rows outside this range bring in
                fixture / pipe-edge noise that confuses the seam.
            max_slope (int): max horizontal pixels of drift per row.
                The gap is near-vertical (max horizontal velocity 2 mm/s
                = ~43 px/s = much less than the vertical strip
                transport rate), so 1 px/row is a generous prior.
            min_mean_likelihood (float): minimum mean per-row matched-
                filter response along the seam for the detection to be
                considered confident. The kernel is normalized by w so
                its response is approximately (mean_shoulder -
                mean_interior) intensity in grey levels — 25 means the
                gap is on average 25 grey-levels darker than its
                shoulders along the path, comfortably below the typical
                ~80 contrast of a clean weld gap and above the ~10
                contrast of background texture.
            max_lateral_excursion (int): max(seam) - min(seam) cap
                across y_min..y_max. A real gap's lateral wander should
                be small. Wandering more than this means the DP is
                stitching together unrelated features.
            search_x_range (tuple): optional (xmin, xmax) restricting
                where the seam may run. Implemented by setting cost to
                +inf outside the range so DP can't enter those columns.
            second_path_ratio (float): minimum ratio between the best
                seam's mean likelihood and the second-best (computed
                after masking out a window around the first). >=1.4
                means the best seam must be >=40% stronger than its
                competitor — anything closer suggests an ambiguous
                bimodal gap distribution and we'd rather reject than
                guess. Wrong predictions cost -10 vs +1 for misses.
            second_path_exclusion (int): pixels +- around the best seam
                masked out of the second-pass DP. Must exceed the gap's
                main-lobe width or the "second" path is just a parallel
                shadow of the first.
            darkness_threshold (float): grey-level at which the absolute-
                darkness weight becomes zero. The true weld gap is by
                far the darkest narrow vertical valley in frame
                (interior typically 70-90 post-CLAHE), so 100 admits
                the gap with margin while suppressing competing
                shallower valleys whose interior sits at 120+. The
                contrast-only matched filter cannot distinguish a deep
                gap from a shallower-but-wider valley because both
                have similar (shoulder - interior); weighting by
                absolute interior darkness breaks that tie.
            max_interior_at_y_line (float): maximum allowed grey-level
                of the seam's pixel at y_line itself. Catches Set 3-style
                frames where the gap is occluded by welding-arc glare
                exactly at y_line — the seam still finds SOME dark path
                across the y_min..y_max strip but at y_line lands on a
                bright glare pixel, so we reject (or extrapolate, see below).
                ~110 is comfortably above any real gap interior and well
                below typical glare/blur intensities (200+).
            extrapolate_when_glared (bool): if True, when the y_line glare
                gate triggers, attempt a line-fit extrapolation from the
                visible portion of the seam (rows below the glare zone)
                instead of immediately rejecting. The gap is approximately
                vertical so a line through the visible (y, x) pairs
                projects back to y_line. DEFAULT FALSE — Set 3 frames
                with reflective (super-glossy) weld centres present a
                parallel non-gap dark feature in the visible region that
                the seam DP locks onto, and extrapolating from THAT path
                produces confidently-wrong answers (-10 each on the
                marking scheme). Returning -1 (-> 0 points) is strictly
                better than guessing wrong on these. Enable explicitly
                only when the visible-region seam is known to track the
                actual gap.
            extrap_min_visible_likelihood (float): per-row likelihood floor
                for a row to be considered "visible" and contribute to the
                extrapolation fit. Excludes rows in the glare zone (where
                likelihood is 0 because no dark valley survives the
                saturation) and rows with weak / partial gap visibility.
            extrap_min_visible_rows (int): minimum number of visible rows
                required before extrapolation is attempted. With too few
                points the line fit is unreliable.
            extrap_max_line_slope (float): maximum |dx/dy| of the fitted
                line. The gap is near-vertical (max horizontal velocity
                2 mm/s vs constant vertical strip transport), so any
                steep fit means the line is tracking unrelated features
                across rows; reject the extrapolation in that case
                because projecting a steep slope ~100 rows back to y_line
                amplifies any error in the slope estimate.
            extrap_ransac_residual (float): RANSAC inlier threshold for
                the line fit, in pixels. Tight (3) so a point >3 px off
                the line is excluded from the fit, which prunes outlier
                rows where the seam wandered onto something other than
                the gap.
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
        vertical-valley kernel.

        Kernel layout (1D, applied row-wise):
            [+1] * shoulder_width   |  [-1] * expected_width   |   [+1] * shoulder_width
                bright shoulder     |     dark gap interior    |       bright shoulder

        Sum of weights = 0 so the response is invariant to global
        brightness offsets (a uniformly bright or dark image responds
        zero everywhere, which is what we want — only the LOCAL contrast
        matters). The output is rectified because only positive values
        (centre darker than shoulders) correspond to dark valleys;
        negative values would mean a bright bump on a dark background,
        which is not a weld gap.

        The kernel is normalised by expected_width so the response is
        on a meaningful scale: numerically ~ (mean_shoulder_intensity
        - mean_interior_intensity). This makes the min_mean_likelihood
        threshold interpretable as a grey-level contrast.

        args:
            img_gray (np.ndarray): single-channel uint8 image
        returns:
            (np.ndarray): float32 likelihood image, same HxW as input.
        """
        img = img_gray.astype(np.float32)
        if self._blur_sigma > 0:
            ksz = max(3, int(2 * round(2 * self._blur_sigma) + 1))
            img = cv2.GaussianBlur(img, (ksz, ksz), self._blur_sigma)

        w = self._expected_width
        s = self._shoulder_width
        kernel = np.concatenate([
            np.ones(s, dtype=np.float32),
            -np.ones(w, dtype=np.float32),
            np.ones(s, dtype=np.float32),
        ]).reshape(1, -1)
        #Normalise so response approximates (shoulder - interior) grey levels.
        # Each side contributes shoulder_width pixels of +1 weight, total 2*s
        # at +1 and w at -1. Dividing by w makes the response read as the
        # interior-darkness-vs-shoulder difference.
        kernel = kernel / float(w)
        contrast_resp = cv2.filter2D(img, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
        contrast_resp = np.maximum(0.0, contrast_resp)

        #Absolute-darkness weighting. The contrast-only matched filter
        # cannot distinguish a deep narrow gap (interior ~75) from a
        # shallower-but-wider parallel valley (interior ~125) because
        # both have similar (shoulder - interior) numerically. The true
        # gap is by far the darkest narrow vertical valley in the FOV,
        # so weighting by absolute interior darkness breaks the tie.
        # mean_interior[y, x] = mean of input over the same w-wide
        # interior column window the matched filter dark-weights at col x.
        interior_kernel = np.concatenate([
            np.zeros(s, dtype=np.float32),
            np.ones(w, dtype=np.float32) / float(w),
            np.zeros(s, dtype=np.float32),
        ]).reshape(1, -1)
        mean_interior = cv2.filter2D(img, cv2.CV_32F, interior_kernel,
                                     borderType=cv2.BORDER_REFLECT)
        #Weight is in [0, 1]: 1 for interior=0 (perfectly dark), 0 for
        # interior >= darkness_threshold. Linear in between.
        darkness_weight = np.clip(
            (self._darkness_threshold - mean_interior) / self._darkness_threshold,
            0.0, 1.0,
        )
        return contrast_resp * darkness_weight

    def _dp_seam(self, cost: np.ndarray) -> tuple:
        """
        Seam-carving-style DP: find the column path from row 0 to row H-1
        of `cost` that minimises the summed cost, with a per-row lateral
        slope limit of self._max_slope. Vectorised across columns —
        Python only loops over the H rows, the inner work is numpy.

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
            #Build a stack of slope-shifted versions of prev. Element
            # (s_idx, x) holds prev[x + (s_idx - slope)], which is the
            # accumulated cost of arriving at row y, column x, having
            # come from column x + (s_idx - slope) at row y-1.
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
            #parent[y, x] = the column in row y-1 from which the best
            # path to (y, x) arrived. Slope offset = best_s_idx - slope,
            # so previous-row column = x + (best_s_idx - slope).
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
        Line-fit extrapolation fallback for frames where the gap is
        occluded by glare at y_line. We trust seam positions only at rows
        where the matched-filter likelihood is above
        extrap_min_visible_likelihood (so glare-zone rows where
        likelihood is ~0 are excluded), then fit a robust line through
        those (y, x) pairs and project to y_line.

        args:
            rows (np.ndarray): absolute y indices the seam covers.
            seam (np.ndarray): seam x positions, same length as rows.
            likelihood_along_seam (np.ndarray): per-row gap-likelihood
                along the seam.
        returns:
            (int) extrapolated x at y_line, or None if extrapolation
                cannot be trusted (insufficient visible rows, slope too
                steep, or RANSAC inlier count too low).
        """
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
            #RANSAC raises if it can't find a consensus set
            return None

        n_inliers = int(np.sum(ransac.inlier_mask_))
        if n_inliers < self._extrap_min_visible_rows:
            return None

        slope = float(ransac.estimator_.coef_[0])
        if abs(slope) > self._extrap_max_line_slope:
            #steep slope would amplify any error in the slope estimate
            # over the ~100-row extrapolation distance to y_line
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
            tuple[int, int]: (position, valid). position = -1 when invalid.
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
        #cost should be float32 for the +inf gating below; .copy() is implicit.
        cost = cost.copy()

        if self._search_x_range is not None:
            xmin, xmax = self._search_x_range
            xmin = max(0, int(xmin))
            xmax = min(W, int(xmax))
            cost[:, :xmin] = np.inf
            cost[:, xmax:] = np.inf

        seam, _total_cost = self._dp_seam(cost)

        #Position at y_line. The seam is indexed by row offset from y0,
        # so the row corresponding to y_line is y_line - y0.
        seam_idx = self._y_line - y0
        if seam_idx < 0 or seam_idx >= len(seam):
            return (-1, 0)
        position = int(seam[seam_idx])

        #Likelihood along the chosen seam (in the absolute-y indexing
        # required to look up likelihood values).
        rows = np.arange(y0, y1)
        likelihood_along_seam = likelihood[rows, seam]
        mean_lk = float(np.mean(likelihood_along_seam))
        excursion = int(seam.max() - seam.min())

        #Second-best seam: re-run DP after masking out a window around
        # the first seam in cost-space. If the second-best path is also
        # very strong, the gap location is ambiguous (twin-valley case).
        masked_cost = cost.copy()
        for y_off, x in enumerate(seam):
            xl = max(0, int(x) - self._second_path_exclusion)
            xr = min(W, int(x) + self._second_path_exclusion + 1)
            masked_cost[y_off, xl:xr] = np.inf
        seam2, _ = self._dp_seam(masked_cost)
        likelihood_along_seam2 = likelihood[rows, seam2]
        mean_lk2 = float(np.mean(likelihood_along_seam2))

        #y_line glare gate: the seam may have found a coherent dark path
        # somewhere in the y_min..y_max strip, but if at y_line specifically
        # the seam sits on a bright pixel the gap is occluded by welding-
        # arc glare exactly where the brief requires the answer (the user-
        # observed Set 3 failure mode). Two options: reject outright, or
        # try line-fit extrapolation from the visible portion of the seam.
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
            reasons.append(f"y_line glare: I[{position},{self._y_line}]={seam_pixel_at_y_line:.0f}>{self._max_interior_at_y_line}")
        if mean_lk < self._min_mean_likelihood and not glare_at_y_line:
            #the mean-lk floor only applies when we trusted the seam at y_line.
            # Under glare the seam in the glared rows has likelihood ~0 by
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
        #When extrapolation succeeded, replace the position with the
        # extrapolated x; valid stays 1 unless another gate failed.
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
        SeamDP-specific interim visualisations. Replaces the base class's
        generic CLAHE-only interim images with detector-relevant outputs:
            1. raw colour ROI — the original image content the detector saw
            2. CLAHE-normalised greyscale ROI — the actual detector input
            3. matched-filter likelihood heatmap of the ROI, with the
               y_line guide and predicted-x line overlaid. The heatmap
               makes the seam DP's reasoning legible: hot pixels are
               where the matched filter says "this is a thin dark
               valley", and the predicted line should sit on a warm column.
               This is the most useful debug artefact when a detection
               looks wrong — you can see immediately whether the matched
               filter even saw the gap or whether some other dark feature
               outshone it.
        """
        x0, x1, y0, y1 = roi
        raw_roi = image[y0:y1, x0:x1].copy()
        gray = self.rgb_to_greyscale(image)
        gray_p = self.preprocess(gray) if self._use_clahe else gray
        clahe_roi = cv2.cvtColor(gray_p[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)

        #Compute likelihood once on the full image then crop, so the
        # filter response is consistent with what the DP actually used —
        # filtering only the ROI would leak boundary effects.
        likelihood = self._gap_likelihood(gray_p)
        like_roi = likelihood[y0:y1, x0:x1]
        #scale to 0..255 with a soft ceiling so a single hot pixel doesn't
        # wash out the rest. Cap at the 99th percentile of the ROI; below
        # that everything maps linearly.
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
        Run the full seam-DP pipeline on a BGR image and return the
        weld-gap position and validity.
        returns:
            tuple[int,int]: (position, valid). position = -1 when invalid.
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
