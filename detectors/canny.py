import cv2
import numpy as np

from detectors.base import WeldDetector, DEBUG


class WeldDetectorCanny(WeldDetector):
    """
    Edge-based weld-gap detector. The gap appears as two near-parallel
    vertical dark edges separated by `expected_width` (~6 px). Detection
    finds those paired edges directly via Gaussian-smoothed Canny on a
    vertical band, rather than correlating against saved templates, so it
    does not depend on the exact appearance of the surface — only on the
    gap geometry. Trades robustness against glare/glints for a stronger
    geometric prior, and complements the template-matching method as a
    sanity-check or fall-back when templates fail to match.
    """
    def __init__(self, y_line:int = 70, expected_width:int = 6, tolerance:int = 3, images=[],
                 canny_low:int = 30, canny_high:int = 90,
                 blur_sigma:float = 0.5, band_height:int = 11,
                 width_tolerance:int = 3, use_clahe:bool = True,
                 min_pair_strength:int = 3,
                 max_interior_intensity:int = 110,
                 min_contrast:float = 20.0,
                 search_x_range:tuple = None):
        """
        args:
            canny_low / canny_high: hysteresis thresholds for Canny.
                Roughly 1:3 ratio per Canny's recommendation, kept low
                because the gap is only ~70 px below its ~170 px shoulders
                — defaults of 50/150 over-suppress one or both edges.
            blur_sigma: pre-Canny Gaussian sigma. Small (~0.5) preserves
                the narrow ~6 px gap; larger values (>=1) smooth one edge
                of the gap into the other and the edge pair vanishes.
                Setting too low (0) lets JPEG blocking and sensor noise
                produce a haze of false edges, so 0.5 is the sweet spot.
            band_height: rows used for vertical edge accumulation. The gap
                is locally vertical for several pixels around y_line, so
                summing edge pixels over a band amplifies the true gap
                signature against random non-vertical edges.
            width_tolerance: +- pixels of slack around expected_width when
                pairing edges. Tighter = stricter geometric prior.
            min_pair_strength: minimum number of band rows in which BOTH
                edges of the pair must be active simultaneously. Acts as a
                confidence floor — a real vertical gap should produce a
                co-occurrence across most rows of the band.
            max_interior_intensity: maximum allowed mean greyscale value
                (post-CLAHE) inside the candidate gap. The gap is by far
                the darkest narrow vertical feature in the FOV — a "gap"
                whose interior is bright cannot be the true weld gap
                regardless of how clean its edge pair looks. ~110 admits
                the true gap (interior usually < 100) while rejecting
                shallow secondary valleys whose interior sits at 150+.
            min_contrast: minimum (mean shoulder intensity - mean interior
                intensity) for the candidate gap. Catches false positives
                where the pair sits in a uniformly dark region (no local
                contrast = no actual gap, just a dark patch). The true gap
                typically has shoulders ~150 vs interior ~70 → contrast ~80.
            search_x_range: optional (xmin, xmax) restricting the columns
                searched. Use to gate the search to a region around a prior
                estimate (e.g. last frame's prediction) and avoid distant
                spatter / bezel edges winning.
        """
        super().__init__(y_line, expected_width, tolerance, images)
        self._canny_low = canny_low
        self._canny_high = canny_high
        self._blur_sigma = blur_sigma
        self._band_height = band_height
        self._width_tolerance = width_tolerance
        self._use_clahe = use_clahe
        self._min_pair_strength = min_pair_strength
        self._max_interior_intensity = max_interior_intensity
        self._min_contrast = min_contrast
        self._search_x_range = search_x_range

    def _edge_band(self, img_gray: np.ndarray, y_center: int) -> tuple:
        """
        Returns (edges, roi) for a horizontal band centred on y_center.
        Smoothing first is essential — differentiation amplifies high-
        frequency noise and without a blur step we get a haze of edges
        from JPEG blocking and sensor noise that drowns the gap. We return
        the smoothed roi alongside the edge map so downstream pair-scoring
        can apply intensity / contrast tests against the same image data
        Canny saw, not the unsmoothed raw greyscale.
        args:
            img_gray (np.ndarray): single-channel uint8 image
            y_center (int): row to centre the band on
        returns:
            (edges, roi): edges is [<=band_height x W] uint8 with 255 where
                Canny fired; roi is the smoothed [<=band_height x W] uint8
                greyscale band used as Canny's input.
        """
        half = self._band_height // 2
        y0 = max(0, y_center - half)
        y1 = min(img_gray.shape[0], y_center + half + 1)
        roi = img_gray[y0:y1, :]

        #Gaussian smoothing prior to Canny. Kernel size derived from sigma
        # (Canny's standard ~6*sigma rule, rounded to odd).
        if self._blur_sigma > 0:
            ksz = max(3, int(2 * round(2 * self._blur_sigma) + 1))
            roi = cv2.GaussianBlur(roi, (ksz, ksz), self._blur_sigma)

        #Canny does its own gradient + non-maximum suppression + hysteresis,
        # so we get thin one-pixel edges directly. This matters: fat edges
        # from a Sobel threshold would inflate the column-projection and
        # confuse the pair-matching width search.
        edges = cv2.Canny(roi, self._canny_low, self._canny_high)
        return edges, roi

    def _find_paired_edges(self, edges_band: np.ndarray, roi_band: np.ndarray) -> tuple:
        """
        Locate the strongest pair of vertical edges separated by
        expected_width +- width_tolerance within the band, validated by
        the intensity profile between the edges.

        Approach:
        1. Project the edge map onto the x-axis: col_strength[x] is the
           number of band rows in which column x is an edge pixel. A real
           vertical gap edge should fire in nearly every row.
        2. Project the smoothed greyscale onto the x-axis: col_intensity[x]
           is the mean intensity of column x across the band. The gap is
           by far the darkest narrow vertical feature in the FOV.
        3. For each candidate width w in [expected_width - tol, +tol] and
           each x, compute three pieces of evidence:
             - pair_strength = min(col_strength[x], col_strength[x+w])
                 — both sides of the gap must be edges (geometric)
             - interior = mean(col_intensity[x+1 : x+w])
                 — the gap interior must be DARK (intensity prior)
             - contrast = mean_shoulders - interior
                 where shoulders = col_intensity sampled symmetrically
                 just outside the candidate gap. The gap should be a
                 LOCAL minimum, not a uniformly dark patch.
        4. Composite score = pair_strength * max(0, contrast). This
           multiplicative form requires BOTH evidence sources to be
           positive — a strong edge pair surrounding a bright region
           scores zero, as does a deep dark valley with no Canny edges.
        5. Among candidates, pick the (x, w) maximising the composite.
        6. Validity gates: pair_strength >= min_pair_strength, interior
           <= max_interior_intensity, contrast >= min_contrast. All three
           must pass — this is the difference between a one-criterion
           Canny detector (the previous version, which locked to the
           shallow secondary valley near image0001's x=951) and a
           three-criterion detector that rejects any candidate failing
           any single test.

        args:
            edges_band (np.ndarray): [Hb x W] uint8 binary edge map
            roi_band (np.ndarray): [Hb x W] uint8 smoothed greyscale band
                (Canny's input — using the smoothed image keeps the
                interior-intensity test consistent with what Canny saw)
        returns:
            tuple[int, int]: (gap_centre, valid). gap_centre = -1 when invalid.
        """
        col_strength = np.sum(edges_band > 0, axis=0).astype(np.int32)        # [W]
        col_intensity = np.mean(roi_band.astype(np.float32), axis=0)          # [W]
        W = col_strength.size

        #optional search-window gating: zero column-strengths outside the
        # allowed range so the argmax cannot pick distant spatter / bezel
        # edges. Intensity isn't masked because we read it positionally
        # by index inside the candidate scan.
        if self._search_x_range is not None:
            xmin, xmax = self._search_x_range
            xmin = max(0, int(xmin))
            xmax = min(W, int(xmax))
            mask = np.zeros_like(col_strength)
            mask[xmin:xmax] = 1
            col_strength = col_strength * mask

        best_composite = -1.0
        best_x = -1
        best_w = self._expected_width
        best_strength = 0
        best_interior = 0.0
        best_contrast = 0.0
        widths = range(max(2, self._expected_width - self._width_tolerance),
                       self._expected_width + self._width_tolerance + 1)
        for w in widths:
            if w >= W:
                continue
            #vectorised pair scan: strength[x] = min(left edge, right edge)
            left = col_strength[:W - w]
            right = col_strength[w:]
            strength = np.minimum(left, right)
            #suppress the trivial peak at the very image edges (camera mount
            # bezels can clip the FOV with high-contrast borders that fire
            # Canny across the whole band)
            strength[:5] = 0
            strength[-5:] = 0

            #interior intensity: mean of the columns strictly between the
            # two edge columns. For w == 2 there is no strictly-interior
            # column so we fall back to the midpoint intensity itself.
            if w >= 3:
                #cumulative-sum trick avoids a Python loop; interior[x] is
                # the mean of col_intensity[x+1 .. x+w-1] inclusive.
                cs = np.cumsum(np.concatenate([[0.0], col_intensity]))
                interior_sum = cs[2 + np.arange(W - w) + (w - 2)] - cs[1 + np.arange(W - w)]
                interior = interior_sum / float(w - 1)
            else:
                interior = col_intensity[np.arange(W - w) + w // 2]

            #contrast: min-of-shoulders intensity (sampled w pixels to
            # the left and w pixels to the right of the candidate) minus
            # interior. We use MIN rather than MEAN because a real gap is
            # a local minimum surrounded by bright pipe surface on BOTH
            # sides — if one shoulder is dim, the candidate is sitting on
            # the edge of a wider dark feature (e.g. the second valley in
            # a multi-feature dim band) rather than being a clean gap.
            # Min-shoulder enforces this symmetry implicitly. Positive =>
            # gap is a real local minimum; negative => the "gap" is
            # brighter than at least one shoulder (impossible for a real
            # weld gap).
            shoulder_w = w
            x_idx = np.arange(W - w)
            left_shoulder_x = np.clip(x_idx - shoulder_w, 0, W - 1)
            right_shoulder_x = np.clip(x_idx + w + shoulder_w, 0, W - 1)
            shoulder = np.minimum(col_intensity[left_shoulder_x], col_intensity[right_shoulder_x])
            contrast = shoulder - interior

            #composite: multiplicative AND of edge-strength and contrast.
            # Either being zero / negative kills the candidate at this x.
            composite = strength.astype(np.float32) * np.maximum(0.0, contrast)

            x = int(np.argmax(composite))
            score = float(composite[x])
            if score > best_composite:
                best_composite = score
                best_x = x
                best_w = w
                best_strength = int(strength[x])
                best_interior = float(interior[x])
                best_contrast = float(contrast[x])

        #Three independent validity gates — wrong detection costs 10x a miss
        # so we'd rather reject than guess.
        if (best_strength < self._min_pair_strength
                or best_interior > self._max_interior_intensity
                or best_contrast < self._min_contrast):
            if DEBUG:
                print(f"  [canny] reject: x={best_x+best_w//2} strength={best_strength} "
                      f"interior={best_interior:.1f} contrast={best_contrast:.1f}")
            return (-1, 0)

        if DEBUG:
            print(f"  [canny] accept: x={best_x+best_w//2} w={best_w} "
                  f"strength={best_strength} interior={best_interior:.1f} contrast={best_contrast:.1f}")
        return (int(best_x + best_w // 2), 1)

    def extract_weld_center_from_band(self, img_gray: np.ndarray, y_center: int) -> tuple:
        """
        2D-band variant of the parent's 1D slice extractor. Pre-processes,
        runs Canny on the band and pairs the dominant vertical edges,
        validated by the intensity profile of the candidate gap interior.
        returns:
            tuple[int, int]: (position, valid). position = -1 when invalid.
        """
        if self._use_clahe:
            img_gray = self.preprocess(img_gray)
        edges, roi = self._edge_band(img_gray, y_center)

        if DEBUG:
            cv2.imwrite("debug_canny_band.png", edges)

        return self._find_paired_edges(edges, roi)

    def process_image(self, image: np.ndarray) -> tuple:
        """
        Run the full Canny pipeline on a BGR image and return the weld-gap
        position and validity.
        returns:
            tuple[int,int]: (position, valid). position = -1 when invalid.
        """
        gray = self.rgb_to_greyscale(img=image)
        position, valid = self.extract_weld_center_from_band(gray, self._y_line)
        image = self.draw_weld(image, position, self._y_line, valid)

        if DEBUG:
            cv2.imshow("Weld Center (Canny)", image)
            cv2.waitKey(0)
        return (position, valid)

    def process_all(self):
        results = []
        for image in self.images:
            results.append(self.process_image(image))
        return results
