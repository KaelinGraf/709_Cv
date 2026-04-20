import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

CWD = os.getcwd()
PIX_WIDTH = 0.04607 #in mm, width of each pixel
MAX_WELD_WIDTH = 0.5 #in mm
MAX_DEV_H = 2.0 #maximum horizontal deviation speed in mm


DEBUG = True


class KalmanTracker1D:
    """
    1D Constant-Velocity Kalman Filter for weld gap position tracking.

    State vector: [position, velocity]^T
    Measurement: position only

    Provides temporal smoothing and prediction-only fallback when
    the CV pipeline flags a frame as invalid (smoke, occlusion, etc).
    The 2 mm/s kinematic limit constrains max inter-frame displacement
    to ~1.44 px/frame at 30 FPS.
    """
    def __init__(self, fps: float = 30.0, process_noise_std: float = 1.0, measurement_noise_std: float = 3.0):
        """
        args:
            fps (float): Camera frame rate in frames per second
            process_noise_std (float): Std dev of process noise in pixels/frame.
                Controls how much the filter trusts the constant-velocity model.
                ~1.0 allows for the 1.44 px/frame kinematic limit with margin.
            measurement_noise_std (float): Std dev of measurement noise in pixels.
                Matches the ±3 pixel accuracy tolerance of the ZNCC pipeline.
        """
        self.dt = 1.0 / fps

        # State vector [position, velocity]
        self.x = np.zeros((2, 1), dtype=np.float64)

        # State covariance matrix — initialized with high uncertainty
        self.P = np.eye(2, dtype=np.float64) * 500.0

        # State transition matrix (constant velocity model)
        # [p_k]   [1  dt] [p_{k-1}]
        # [v_k] = [0   1] [v_{k-1}]
        self.F = np.array([[1.0, self.dt],
                           [0.0, 1.0]], dtype=np.float64)

        # Measurement matrix — we only observe position
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)

        # Process noise covariance Q
        # Discrete white noise model for constant velocity
        q = process_noise_std ** 2
        self.Q = q * np.array([[self.dt**3 / 3.0, self.dt**2 / 2.0],
                                [self.dt**2 / 2.0, self.dt]], dtype=np.float64)

        # Measurement noise covariance R
        self.R = np.array([[measurement_noise_std ** 2]], dtype=np.float64)

        self._initialized = False

    def predict(self) -> float:
        """
        Kalman predict step. Advances state estimate by one time step.
        Returns predicted position.
        """
        if not self._initialized:
            return -1.0

        # x_k|k-1 = F * x_{k-1|k-1}
        self.x = self.F @ self.x
        # P_k|k-1 = F * P_{k-1|k-1} * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        return float(self.x[0, 0])

    def update(self, measurement: float):
        """
        Kalman update step. Incorporates a new position measurement.
        args:
            measurement (float): Measured weld gap position in pixels
        """
        if not self._initialized:
            # First valid measurement initializes the filter state
            self.x[0, 0] = measurement
            self.x[1, 0] = 0.0  # Assume zero initial velocity
            self.P = np.eye(2, dtype=np.float64) * 10.0
            self._initialized = True
            return

        z = np.array([[measurement]], dtype=np.float64)

        # Innovation (measurement residual)
        y = z - self.H @ self.x
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Updated state estimate
        self.x = self.x + K @ y
        # Updated covariance (Joseph form for numerical stability)
        I = np.eye(2, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

    def get_position(self) -> float:
        """Returns the current estimated position."""
        if not self._initialized:
            return -1.0
        return float(self.x[0, 0])

    def get_velocity(self) -> float:
        """Returns the current estimated velocity in pixels/frame."""
        if not self._initialized:
            return 0.0
        return float(self.x[1, 0])

    def get_gating_window(self, n_sigma: float = 5.0) -> tuple:
        """
        Returns a (low, high) pixel range for ROI gating based on
        the predicted position ± n_sigma * position_std.
        This restricts the search window for the next frame, preventing
        the CV algorithm from jumping to distant spatter artifacts.
        """
        if not self._initialized:
            return None
        pos = float(self.x[0, 0])
        pos_std = float(np.sqrt(self.P[0, 0]))
        half_window = n_sigma * pos_std
        return (int(max(0, pos - half_window)), int(pos + half_window))


class WeldDetector:
    def __init__(self, y_line: int = 70, expected_width: int = 6, tolerance: int = 3,
                 templates=[], images=[],
                 psr_threshold: float = 10.0, pspr_threshold: float = 1.5,
                 fps: float = 30.0, peak_exclusion_window: int = 5):
        """
        args:
            y_line (int): Line of expected weld center location in pixels
            expected_width (int): expected weld width in pixels
            tolerance (int): tolerance of prediction
            templates([str]): list of relative paths to template npy files
            psr_threshold (float): Minimum Peak-to-Sidelobe Ratio for valid detection.
                A sharp, distinct peak above background noise indicates high confidence.
            pspr_threshold (float): Minimum Peak-to-Second-Peak Ratio for valid detection.
                Ensures the primary peak is uniquely dominant (rejects twin artifacts).
            fps (float): Camera frame rate for Kalman filter temporal model
            peak_exclusion_window (int): Pixels ± around peak to exclude from sidelobe/second-peak calculation
        """
        self._y_line: int = y_line
        self._expected_width: int = expected_width
        self._tolerance: int = tolerance
        self._psr_threshold: float = psr_threshold
        self._pspr_threshold: float = pspr_threshold
        self._peak_exclusion: int = peak_exclusion_window
        self.templates = []
        self.images = []
        self.kalman = KalmanTracker1D(fps=fps)
        for image in images:
            self.images.append(self.load_image(image))
        self.load_templates(templates=templates)

        
    def load_image(self,path:str=None):
    
        if path is not None:
            img = cv2.imread(os.path.join(CWD,path))
            if img is None:
                raise Exception("File could not be loaded")

        return img #np array
    def crop_to_roi(self,img,height:int,width:int,x_center:int):
        """
        Crops image to ROI around weld center line
        args:
            img(np.array): image loaded by opencv
            height(int): height of crop (pixels). Should be divisible by two
            width(int): width of crop (pixels). Should be divisible by two
            x_center(int):approximate x location of the weld (pixels)
        """
        if height%2 != 0 or width%2!=0:
            raise Exception("ROI sizes not divisible by two")
        if height/2 > self._y_line:
            raise Exception(f"ROI height exceeds image bounds, must be less than {self._y_line*2}")
        w,h = img.size

        return None

    def rgb_to_greyscale(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    def crop_to_weld_line(self,img_gray:np.ndarray,y_line:int) -> np.ndarray:
        """
        crop image to weld line to extract approximate ROI
        args:
            img_gray(np.ndarray): greyscale image
            y_line(int): line of the array (from the top) to extract from

        returns:
            weld_line_slice(np.ndarray): Horizontal image slice taken at y_line 
        """
        return img_gray[y_line, :]

    def _compute_psr(self, signal: np.ndarray, peak_idx: int) -> float:
        """
        Compute Peak-to-Sidelobe Ratio (PSR).
        PSR = (P_max - mu_sidelobe) / sigma_sidelobe

        Measures how sharply the primary peak stands out above the background
        noise floor. Low PSR indicates the signal is degraded (e.g. smoke
        occlusion, severe glare) and detection is unreliable.

        args:
            signal (np.ndarray): The combined MoE probability distribution
            peak_idx (int): Index of the primary peak
        returns:
            psr (float): Peak-to-Sidelobe Ratio
        """
        mask = np.ones(len(signal), dtype=bool)
        low = max(0, peak_idx - self._peak_exclusion)
        high = min(len(signal), peak_idx + self._peak_exclusion + 1)
        mask[low:high] = False

        sidelobe = signal[mask]
        if len(sidelobe) == 0 or np.std(sidelobe) == 0:
            return 0.0

        mu_sl = np.mean(sidelobe)
        sigma_sl = np.std(sidelobe)

        return (signal[peak_idx] - mu_sl) / sigma_sl

    def _compute_pspr(self, signal: np.ndarray, peak_idx: int) -> float:
        """
        Compute Peak-to-Second-Peak Ratio (PSPR).
        PSPR = P_max / P_2nd_peak

        Measures uniqueness of the primary peak. If PSPR ≈ 1.0, multiple
        peaks have nearly identical strength (e.g. parallel scratches),
        making the true weld gap ambiguous. High PSPR means the primary
        peak is unequivocally dominant.

        args:
            signal (np.ndarray): The combined MoE probability distribution
            peak_idx (int): Index of the primary peak
        returns:
            pspr (float): Peak-to-Second-Peak Ratio
        """
        mask = np.ones(len(signal), dtype=bool)
        low = max(0, peak_idx - self._peak_exclusion)
        high = min(len(signal), peak_idx + self._peak_exclusion + 1)
        mask[low:high] = False

        sidelobe = signal[mask]
        if len(sidelobe) == 0:
            return float('inf')

        second_peak = np.max(sidelobe)
        if second_peak == 0:
            return float('inf')

        return signal[peak_idx] / second_peak

    def extract_weld_center_from_slice(self, weld_line_slice: np.ndarray) -> tuple:
        """
        Extracts the pixel location of the weld center using 1D ZNCC template
        matching with Mixture of Experts (MoE) fusion and PSR/PSPR confidence
        validation.

        Pipeline:
        1. Per-template ZNCC cross-correlation
        2. Rectification (clamp negatives to zero)
        3. Square to suppress low-level noise
        4. Normalize each into a PDF
        5. Fuse all PDFs via additive Mixture of Experts (arithmetic mean)
        6. Compute PSR and PSPR on the combined distribution
        7. If both metrics pass thresholds, return (argmax_position, 1)
           Otherwise return (-1, 0) to trigger fail-safe

        returns:
            tuple[int, int]: (position, valid) where valid ∈ {0, 1}
                position: pixel coordinate of weld center, or -1 if invalid
                valid: 1 if detection is confident, 0 if fail-safe triggered
        """
        pdfs = []
        if self.templates.size == 0:
            raise Exception("Templates not found, please run self.load_templates")

        for template in self.templates:
            # Perform zero-mean normalized cross-correlation
            res = cv2.matchTemplate(weld_line_slice, template, cv2.TM_CCOEFF_NORMED)
            res = res.flatten()
            # Rectify: negative correlation is the inverse of the gradient we seek
            res_rectified = np.maximum(0, res)
            # Square to emphasize strong matches and suppress low-level noise
            res_filtered = res_rectified ** 2
            # Normalize into probability distribution function
            area = np.sum(res_filtered)
            if area > 0:
                pdf = res_filtered / area
                pdfs.append(pdf)

        if len(pdfs) == 0:
            return (-1, 0)

        # Cast to array
        pdfs = np.array(pdfs)

        if DEBUG:
            plt.figure(figsize=(12, 6))

            # Use a colormap to give each template a unique color
            colors = plt.cm.tab10(np.linspace(0, 1, len(pdfs)))

            for i, pdf in enumerate(pdfs):
                color = colors[i % len(colors)]
                # Plot the distribution curve
                plt.plot(pdf, label=f"Template {i} PDF", color=color, alpha=0.7)

            plt.title("Weld Template Probability Distributions (Overlaid)")
            plt.xlabel("Pixel Coordinate (Shifted by Template Width)")
            plt.ylabel("Probability")
            # Place legend outside the plot so it doesn't cover the peaks
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("debug_templates.png")
            plt.close()

        # --- Mixture of Experts (MoE) Fusion ---
        # Additive combination acts as logical OR: consensus from the majority
        # of templates dominates. A single dissenting template cannot veto the
        # collective agreement, unlike the fragile Product of Experts (PoE).
        final_pdf = np.mean(pdfs, axis=0)
        total = np.sum(final_pdf)
        if total == 0:
            return (-1, 0)
        final_pdf = final_pdf / total

        # --- Peak Detection ---
        peak_idx = int(np.argmax(final_pdf))

        # --- Confidence Validation: PSR & PSPR ---
        # PSR guarantees peak sharpness (signal-to-noise).
        # PSPR guarantees peak uniqueness (rejects twin artifacts like parallel scratches).
        # Both must pass for the measurement to be considered valid.
        psr = self._compute_psr(final_pdf, peak_idx)
        pspr = self._compute_pspr(final_pdf, peak_idx)

        valid = 1 if (psr >= self._psr_threshold and pspr >= self._pspr_threshold) else 0
        position = peak_idx if valid else -1

        if DEBUG:
            status = "VALID" if valid else "INVALID"
            plt.figure(figsize=(12, 6))
            plt.plot(final_pdf)
            plt.title(f"Final MoE Distribution | PSR={psr:.2f}  PSPR={pspr:.2f}  [{status}]")
            plt.xlabel("Pixel Coordinate (Shifted by Template Width)")
            plt.ylabel("Probability")
            if valid:
                plt.axvline(x=peak_idx, color='g', linestyle='--', label=f'Peak @ {peak_idx}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig("debug_final.png")
            plt.close()

        return (position, valid)

    def load_templates(self, template_dir="templates", templates=None, load_all=True):
        """
        Loads templates from either specified paths, or from directory search (or both if load_all is True)
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
            for file in files_set:
                self.templates.append(np.load(file)[0])
        else:
            if templates:
                files_list = set(templates)
            else:
                files_list = set(dir_files)
            for file in files_list:
                self.templates.append(np.load(file)[0])
        
        self.templates = np.asarray(self.templates)
        print(f"Loaded {len(self.templates)} templates.")

    def extract_weld_center_line_fitting(self,image_greyscale: np.ndarray,eval_lines: np.ndarray, linefit_algo = "RANSAC"):
        """
        Extracts the pixel location of the weld center via performing statistical (cross correlation based) prediction at multiple y-lines, then performing outlier-robust line fitting to regress weld location at y-line (70)
        args:
            image_greyscale (np.ndarray): [h x w x 1] image (call rgb_to_greyscale on input image)
            eval_lines (np.ndarray): lines to evaluate weld center for line fitting. More is better, and a wide range is ideal to fit a more accurate line 
            linefit_algo (str): Algorithm selection for line-fitting. NOTE: FILL IN DETAILS OF EACH ALGO HERE ONCE IMPLIMENTED 
        returns:
            list of tuples: [(position, valid), ...] for each eval_line
        """
        algos = ["RANSAC", "LINEAR", "WLEASTSQUARES"]
        if linefit_algo not in algos: raise Exception(f"Linefit algo invalid, choose from {algos}")
        if any(eval_lines>image_greyscale.shape[1]): raise Exception(f"target eval line exceeds image dimensions of {image_greyscale.shape[0]} x {image_greyscale.shape[1]}")
        results = []
        for eval_line in eval_lines:
            weld_crop = self.crop_to_weld_line(image_greyscale,int(eval_line))
            result = self.extract_weld_center_from_slice(weld_line_slice=weld_crop)
            results.append(result)

        print(f"pred centers = {results}")
        return results

        
    def pix_to_world(self):
        pass
    def draw_weld(self, image: np.ndarray, predicted_center: int, y_line: int = 70,
                  valid: int = 1, source: int = 0) -> np.ndarray:
        """
        Draws a marker at the predicted weld center, color-coded by detection state:
            Green  = valid, measured
            Yellow = valid, Kalman-predicted (interpolating through occlusion)
            Red    = invalid (fail-safe triggered, holds position)
        args:
            image (np.ndarray): The image to draw on
            predicted_center (int): The predicted x coordinate of the weld center
            y_line (int): The y coordinate for the marker
            valid (int): 1 if detection is confident, 0 if fail-safe
            source (int): 0 = measured, 1 = kalman predicted
        returns:
            image (np.ndarray): The image with the marker drawn
        """
        if valid == 0 or predicted_center < 0:
            # Draw a red X at the last known position to indicate fail-safe
            return image

        if source == 0:
            color = (0, 255, 0)   # Green — direct measurement
        else:
            color = (0, 255, 255) # Yellow — Kalman prediction (interpolating)

        cv2.drawMarker(image, (int(predicted_center), y_line), color,
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        return image

    def process_image(self, image: np.ndarray) -> tuple:
        """
        Process a single frame through the full pipeline:
        1. Extract weld center from 1D slice via MoE + PSR/PSPR
        2. Feed into Kalman filter for temporal smoothing
        3. Draw visualization marker

        This method is STATEFUL — the Kalman filter maintains state across
        sequential calls, as required for the real-time control loop where
        each call represents one frame from the camera.

        returns:
            tuple[int, int, int]: (filtered_position, valid, source)
                filtered_position: Kalman-filtered pixel coordinate, or -1
                valid: 1 if position is usable, 0 if total fail-safe
                source: 0 = measured, 1 = Kalman predicted (during occlusion)
        """
        eval_lines=np.array([70,90,50,200])
        results = self.extract_weld_center_line_fitting(self.rgb_to_greyscale(img=image),eval_lines=eval_lines)

        # Use the primary eval line result
        raw_position, raw_valid = results[0]

        # --- Kalman Filter Integration ---
        predicted_pos = self.kalman.predict()

        if raw_valid == 1 and raw_position >= 0:
            # Valid measurement: update Kalman with measured position
            self.kalman.update(float(raw_position))
            filtered_position = int(round(self.kalman.get_position()))
            valid = 1
            source = 0  # Measured
        elif self.kalman._initialized:
            # Invalid measurement but Kalman has prior state:
            # use prediction-only (interpolate through occlusion)
            filtered_position = int(round(predicted_pos))
            valid = 1
            source = 1  # Kalman predicted
        else:
            # No valid measurement and Kalman not yet initialized
            filtered_position = -1
            valid = 0
            source = 0

        # Draw visualization
        for i, (pos, v) in enumerate(results):
            self.draw_weld(image, filtered_position if i == 0 else pos,
                          int(eval_lines[i]), valid, source)

        cv2.imshow("Weld Center", image)
        cv2.waitKey(0)

        return (filtered_position, valid, source)

    def process_all(self):
        for image in self.images:
            self.process_image(image)


def main():
    images = ['WeldGapImages_export/Set 1/image0001.jpg']
    weld_detector = WeldDetector(images = images)
    weld_detector.process_all()




if __name__ == "__main__":
    main()