import cv2
import numpy as np
from sklearn import linear_model
import os
import matplotlib.pyplot as plt

from detectors.base import WeldDetector, CWD, DEBUG


class WeldDetectorTemplateMatching(WeldDetector):
    def __init__(self, y_line=70, expected_width=6, tolerance=3,
                 templates=[], images=[],
                 psr_threshold:float = 3.5, peak_exclusion:int = 30,
                 band_height:int = 5, use_clahe:bool = True,
                 min_inliers:int = 3, ransac_residual_threshold:float = 4.0,
                 agreement_tolerance:int = 4, min_agreement_frac:float = 0.27,
                 per_template_blur_sigma:float = 2.0,
                 search_x_range:tuple = None):
        """
        args:
            psr_threshold (float): minimum Peak-to-Sidelobe Ratio for the
                fused MoE distribution. Below this the response is mush
                (smoke / glare frame) and we shouldn't trust any peak.
                Kept as a coarse sanity check; the fine confidence test is
                template-vote agreement.
            peak_exclusion (int): pixels +- around the primary peak masked
                out when computing sidelobe stats — must exceed the main
                correlation lobe width (~ template_width) or the PSR
                statistic self-cancels by including the lobe in the noise.
            band_height (int): rows averaged at each y-line for noise
                reduction. Set to 1 to mimic the original single-row sample.
            use_clahe (bool): apply CLAHE to the greyscale image before
                extraction.
            min_inliers (int): minimum number of confident eval-line
                measurements required before the regression is trusted at
                y_line. Below this we return invalid rather than guess.
            ransac_residual_threshold (float): max residual (px) allowed by
                RANSAC to count an eval-line as an inlier. Approximately the
                ZNCC pipeline's per-line accuracy.
            agreement_tolerance (int): pixels +- around the fused MoE peak
                within which an individual template's argmax counts as
                "agreeing". Should be a few px wider than the +- 3 marking
                tolerance so we don't reject valid detections that drift
                within tolerance.
            min_agreement_frac (float): minimum fraction of templates whose
                individual argmax must agree with the fused peak for the
                detection to be considered confident. Templates were sampled
                from genuine gaps, so a real gap should attract a clear
                majority. Distractor features (bezels, parallel seams) tend
                to split the templates' votes.
            per_template_blur_sigma (float): Gaussian sigma used to smooth
                each per-template correlation response before taking its
                argmax. Suppresses single-pixel noise spikes that would
                otherwise jitter the per-template argmax off the true lobe.
            search_x_range (tuple): optional (xmin, xmax) restricting the
                columns considered. Use to gate the search to a region
                around a prior estimate (e.g. last frame's prediction) and
                avoid distant fixture / bezel features attracting templates.
        """
        super().__init__(y_line, expected_width, tolerance, images)
        self.templates = []
        self._psr_threshold = psr_threshold
        self._peak_exclusion = peak_exclusion
        self._band_height = band_height
        self._use_clahe = use_clahe
        self._min_inliers = min_inliers
        self._ransac_residual_threshold = ransac_residual_threshold
        self._agreement_tolerance = agreement_tolerance
        self._min_agreement_frac = min_agreement_frac
        self._per_template_blur_sigma = per_template_blur_sigma
        self._search_x_range = search_x_range
        self.load_templates(templates=templates)

    def _compute_psr(self, signal: np.ndarray, peak_idx: int) -> float:
        """
        Peak-to-Sidelobe Ratio: (peak - mean_sidelobe) / std_sidelobe.
        High = the primary peak protrudes well above the background noise
        floor. Low = the response is mush, e.g. a smoke / glare frame in
        which no template gets a clean lock.
        """
        mask = np.ones(len(signal), dtype=bool)
        low = max(0, peak_idx - self._peak_exclusion)
        high = min(len(signal), peak_idx + self._peak_exclusion + 1)
        mask[low:high] = False
        sidelobe = signal[mask]
        if len(sidelobe) == 0 or np.std(sidelobe) == 0:
            return 0.0
        return float((signal[peak_idx] - np.mean(sidelobe)) / np.std(sidelobe))

    def _template_agreement(self, per_template_argmax: np.ndarray, fused_peak: int) -> float:
        """
        Fraction of templates whose individual argmax falls within
        agreement_tolerance pixels of the fused MoE peak.

        Replaces PSPR (Peak-to-Second-Peak Ratio) as the fine-grained
        confidence test. PSPR was unreliable here because the wide-FOV
        camera contains several dark vertical features (bezel, fixture,
        parallel seams) that produce strong false correlation peaks ~as
        bright as the true gap — PSPR ~ 1 even when the fused argmax is
        correct. Template-vote agreement is more discriminative: templates
        sampled from genuine gaps preferentially align on the true gap, and
        their per-template argmaxes scatter on noisy / ambiguous frames.
        args:
            per_template_argmax (np.ndarray): each template's argmax, in
                gap-centre image coordinates (already shifted by template_width//2).
            fused_peak (int): the MoE-fused argmax, also in gap-centre coords.
        returns:
            (float): fraction in [0, 1].
        """
        if len(per_template_argmax) == 0:
            return 0.0
        within = np.abs(per_template_argmax - fused_peak) <= self._agreement_tolerance
        return float(np.sum(within)) / float(len(per_template_argmax))

    def extract_weld_center_from_slice(self,weld_line_slice:np.ndarray) -> tuple:
        """
        Extracts the pixel location of the weld center by treating the slice
        as a 1D signal and performing zero-mean normalized cross-correlation
        against each saved gap template. Per-template responses are fused
        via Mixture-of-Experts; the fused peak is validated by template-vote
        agreement (per-template argmaxes clustering near the fused peak)
        and a coarse PSR sanity check.

        Pipeline:
        1. Per-template ZNCC (TM_CCOEFF_NORMED) — DC-bias and scale invariant
        2. Rectify (negative correlation = an INVERTED gap profile, which is
           not what we want to fuse with the positive gap matches)
        3. Square the rectified signal to suppress small spurious correlations
           and emphasise real matches
        4. Smooth each per-template response with a small Gaussian to
           stabilise its argmax against single-pixel noise spikes (without
           shifting the lobe centre)
        5. Optionally mask the response outside search_x_range so distant
           fixture / bezel features can't attract any template
        6. Normalize each per-template response into a PDF
        7. Mixture-of-Experts fusion: arithmetic mean across templates.
           Replaces the previous Product-of-Experts (log-product) fusion,
           which acted as a black-hole veto: a single template hitting near
           zero at the true location killed the combined response. MoE lets
           the majority of templates outvote individual misses.
        8. Compute fused argmax. Compute per-template argmaxes; keep
           detection only if a min_agreement_frac fraction of templates
           individually peak within agreement_tolerance of the fused peak.
           This is the primary confidence check — wrong detections cost
           10x a missed one (-10 vs +1 in marking) so unsure frames MUST
           return invalid.
        9. Coarse PSR sanity check on the fused PDF — if even the fused
           peak doesn't protrude above the noise floor the response is mush
           (smoke / heavy glare) and we reject regardless of agreement.
        10. Shift the argmax by template_width//2 because matchTemplate's
            output index k corresponds to the template's LEFT EDGE at column
            k of the input. Our templates are sampled with the gap centred
            (template_extractor.py uses x +- template_width//2) so the gap
            sits at template index template_width//2 — adding that shift
            maps the correlation index back to the gap centre in image space.

        returns:
            tuple[int, int]: (position, valid). position = -1 when invalid.
        """
        if self.templates.size == 0:
            raise Exception("Templates not found, please run self.load_templates")

        per_template_argmax_x = []  #per-template gap-centre votes in IMAGE-x coords
        per_template_responses = []  #raw smoothed responses, for fused PSR sanity check

        for template, t_offset in zip(self.templates, self.template_offsets):
            #Perform zero-mean normalized cross-correlation
            res = cv2.matchTemplate(weld_line_slice,template,cv2.TM_CCOEFF_NORMED)
            res = res.flatten()
            #rectify and square — only positive correlation peaks correspond to a gap match
            res_rectified = np.maximum(0,res)
            res_filtered = res_rectified ** 2

            #Smooth with a small Gaussian so the per-template argmax sits at
            # the lobe centre rather than the noisiest pixel within it.
            # Width-1 reshape keeps cv2.GaussianBlur happy on a 1D signal.
            if self._per_template_blur_sigma > 0:
                ksz = max(3, int(2 * round(2 * self._per_template_blur_sigma) + 1))
                res_filtered = cv2.GaussianBlur(res_filtered.reshape(1,-1).astype(np.float32),
                                                (ksz, 1),
                                                self._per_template_blur_sigma).flatten()

            #Search-range gating in correlation-output space. matchTemplate
            # output index k corresponds to image x = k + t_offset for THIS
            # template, so we translate the user-supplied image-x range into
            # this template's k-range and zero out everything else.
            if self._search_x_range is not None:
                xmin, xmax = self._search_x_range
                kmin = max(0, int(xmin) - int(t_offset))
                kmax = max(0, int(xmax) - int(t_offset))
                gated = np.zeros_like(res_filtered)
                gated[kmin:kmax] = res_filtered[kmin:kmax]
                res_filtered = gated

            if np.max(res_filtered) <= 0:
                continue
            #convert to image-x: argmax k -> gap centre at k + t_offset
            per_template_argmax_x.append(int(np.argmax(res_filtered)) + int(t_offset))
            per_template_responses.append(res_filtered)

        if len(per_template_argmax_x) == 0:
            return (-1, 0)

        per_template_argmax_x = np.array(per_template_argmax_x)

        #Robust position estimate via median of per-template argmaxes. Median
        # is unaffected by up to (N-1)/2 outlier templates locking onto
        # distractors, AND tolerates the natural spread of votes around the
        # true gap (templates with internal offsets ranging 2..8 produce
        # votes that differ by up to ~6 px even when all are correctly
        # locked on the same gap). Cluster / mean-shift approaches are too
        # brittle here because they require a tight pile-up that this spread
        # prevents — the median absorbs the spread for free.
        position = int(np.median(per_template_argmax_x))

        #Support count: how many templates vote within agreement_tolerance
        # of the median. This is the per-line confidence metric — a real
        # gap should attract a clear majority of templates within a small
        # window of the median, while distractor-dominated frames produce
        # scattered votes with little pile-up at any one location.
        within = np.abs(per_template_argmax_x - position) <= self._agreement_tolerance
        agreement = float(np.sum(within)) / float(len(per_template_argmax_x))
        #Refine position to the median of just the agreeing votes — drops
        # any far outlier still pulling the global median by a pixel or two.
        if np.sum(within) > 0:
            position = int(np.median(per_template_argmax_x[within]))

        #PSR on the fused MoE distribution as a coarse signal-vs-mush check.
        # We work in correlation-output (k) space rather than image-x because
        # different templates have different t_offsets — the fused PDF can't
        # be directly compared across templates without aligning, and PSR is
        # only used as a noise-floor check rather than for localisation.
        fused = np.mean(np.array(per_template_responses), axis=0)
        if fused.sum() > 0:
            fused = fused / fused.sum()
            psr = self._compute_psr(fused, int(np.argmax(fused)))
        else:
            psr = 0.0

        if DEBUG:
            plt.figure(figsize=(12,6))
            plt.plot(fused)
            for am in per_template_argmax_x:
                #plot in k-space using the average offset, since axes are k
                plt.axvline(x=am - int(np.mean(self.template_offsets)),
                            color='orange', linestyle=':', alpha=0.4)
            plt.title(f"Fused response | PSR={psr:.2f}  agreement={agreement:.2f}  cluster_x={position}")
            plt.xlabel("Pixel Coordinate (template-edge / k-space)")
            plt.ylabel("Probability")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("debug_final.png")
            plt.close()

        if psr < self._psr_threshold or agreement < self._min_agreement_frac:
            return (-1, 0)

        return (int(position), 1)

    def load_templates(self, template_dir="templates", templates=None, load_all=True):
        """
        Loads templates from either specified paths, or from directory search (or both if load_all is True).
        Also computes the per-template intra-template gap offset — the index
        of the template's darkest pixel — because templates were sampled by
        manual clicks that weren't always perfectly centred on the gap, so
        the gap sits at a slightly different internal index in each template
        (observed range 2..9 across the saved set). matchTemplate's output
        index k corresponds to the template's left edge in the input, so to
        convert k to gap-centre image x we must add THIS template's gap
        index, not template_width//2.
        args:
            templates [str]: Local/relative path directories assuming CWD is the root
            load_all [bool]: Load all files (list and directory)
        """
        if templates is None:
            templates = []

        full_template_dir = os.path.join(CWD, template_dir)
        if not os.path.exists(full_template_dir):
            os.makedirs(full_template_dir, exist_ok=True)

        # Get all files inside template_dir and prefix them so np.load can find them
        dir_files = [os.path.join(template_dir, file) for file in os.listdir(full_template_dir) if "template" in file and ".npy" in file]

        if load_all:
            files_set = set(dir_files)
            if templates:
                files_set.update(templates)
            #sorted() so loading is deterministic across runs — set() iteration
            # order varies, which made downstream behaviour (which template wins
            # ties, which template's response is used in any order-sensitive
            # downstream step) non-reproducible between runs.
            for file in sorted(files_set):
                self.templates.append(np.load(file)[0])
        else:
            if templates:
                files_list = set(templates)
            else:
                files_list = set(dir_files)
            for file in sorted(files_list):
                self.templates.append(np.load(file)[0])

        self.templates = np.asarray(self.templates)
        #per-template gap offset: argmin within the template = the dark gap
        # pixel. Used to correct the matchTemplate index k -> gap centre x.
        self.template_offsets = np.array([int(np.argmin(t)) for t in self.templates])
        print(f"Loaded {len(self.templates)} templates. Internal gap offsets: {self.template_offsets.tolist()}")

    def extract_weld_center_line_fitting(self,image_greyscale: np.ndarray,eval_lines: np.ndarray, linefit_algo = "RANSAC") -> tuple:
        """
        Extracts the pixel location of the weld center via per-y-line ZNCC
        prediction followed by outlier-robust regression to interpolate the
        gap position at y_line. Only eval-lines whose per-line PSR/PSPR vote
        them confident contribute to the fit — ambiguous lines are dropped
        rather than allowed to drag the regression towards spatter / glare
        artefacts as in the original implementation.

        args:
            image_greyscale (np.ndarray): [h x w] greyscale image
            eval_lines (np.ndarray): y-coordinates to evaluate. More is
                better, and a wide range improves slope accuracy. They must
                straddle y_line for an interpolation rather than an
                extrapolation.
            linefit_algo (str): "RANSAC" (default, outlier-robust),
                "LINEAR" (ordinary least squares) or "WLEASTSQUARES"
                (Ridge regression — adds a small L2 penalty that biases the
                slope towards zero, helpful when the gap is near-vertical).
        returns:
            tuple[int, int]: (position_at_y_line, valid)
        """
        algos = ["RANSAC", "LINEAR", "WLEASTSQUARES"]
        if linefit_algo not in algos: raise Exception(f"Linefit algo invalid, choose from {algos}")
        if any(eval_lines >= image_greyscale.shape[0]):
            raise Exception(f"target eval line exceeds image height of {image_greyscale.shape[0]}")

        #pre-process once for the whole image so every per-line band sees the
        # same CLAHE-normalised greyscale (cheap; CLAHE is O(n) in pixels)
        if self._use_clahe:
            image_greyscale = self.preprocess(image_greyscale)

        confident_x = []
        confident_y = []
        for eval_line in eval_lines:
            band = self.crop_to_weld_band(image_greyscale, int(eval_line), self._band_height)
            position, valid = self.extract_weld_center_from_slice(band)
            if valid == 1:
                confident_x.append(position)
                confident_y.append(int(eval_line))

        if DEBUG:
            print(f"confident eval-line predictions: {list(zip(confident_y, confident_x))}")

        # Need at least min_inliers confident lines to trust the fit. 1-2 is
        # a degenerate fit (any 2 points define a line) so we'd be unable to
        # detect outliers; below the floor we return invalid.
        if len(confident_x) < self._min_inliers:
            return (-1, 0)

        confident_x = np.array(confident_x, dtype=np.float64)
        confident_y = np.array(confident_y, dtype=np.float64).reshape(-1, 1)

        if linefit_algo == "RANSAC":
            #residual_threshold tied to the per-line ZNCC accuracy. min_samples=2
            # is the minimum for a line.
            ransac = linear_model.RANSACRegressor(min_samples=2,
                                                  residual_threshold=self._ransac_residual_threshold)
            ransac.fit(confident_y, confident_x)
            #Reject the frame if RANSAC ended up with too few inliers — a
            # weak consensus across rows usually means a wandering false peak.
            n_inliers = int(np.sum(ransac.inlier_mask_))
            if n_inliers < self._min_inliers:
                return (-1, 0)
            prediction = ransac.predict(np.array([[self._y_line]]))[0]
        elif linefit_algo == "LINEAR":
            linear = linear_model.LinearRegression()
            linear.fit(confident_y, confident_x)
            prediction = linear.predict(np.array([[self._y_line]]))[0]
        elif linefit_algo == "WLEASTSQUARES":
            wleast_squares = linear_model.Ridge()
            wleast_squares.fit(confident_y, confident_x)
            prediction = wleast_squares.predict(np.array([[self._y_line]]))[0]

        return (int(round(prediction)), 1)



    def process_image(self,image:np.ndarray) -> tuple:
        """
        Run the full template-matching pipeline on a BGR image and return
        the weld-gap position and validity.
        returns:
            tuple[int,int]: (position, valid). position = -1 when invalid.
        """
        #Narrow symmetric range straddling y_line. The gap is most reliably
        # visible exactly at the welding-torch focal area (~y_line); rows
        # further away start picking up surrounding fixture / pipe-edge
        # features that compete with the gap. Using a tight band keeps the
        # per-line median honest while still giving RANSAC several
        # measurements to reject outliers across.
        eval_lines = np.arange(max(0, self._y_line - 25), self._y_line + 25 + 1, max(self._band_height, 5))
        position, valid = self.extract_weld_center_line_fitting(self.rgb_to_greyscale(img=image), eval_lines=eval_lines, linefit_algo="RANSAC")
        image = self.draw_weld(image, position, self._y_line, valid)

        if DEBUG:
            cv2.imshow("Weld Center", image)
            cv2.waitKey(0)
        return (position, valid)

    def process_all(self):
        results = []
        for image in self.images:
            results.append(self.process_image(image))
        return results
