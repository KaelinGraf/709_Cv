import cv2
import numpy as np
import os

CWD = os.getcwd()
PIX_WIDTH = 0.04607 #in mm, width of each pixel
MAX_WELD_WIDTH = 0.5 #in mm
MAX_DEV_H = 2.0 #maximum horizontal deviation speed in mm


DEBUG = False


class WeldDetector:
    def __init__(self,y_line:int=70, expected_width:int = 6, tolerance:int = 3, images = []):
        """
        args:
            y_line (int): Line of expected weld center location in pixels
            expected_width (int): expected weld width in pixels
            tolerance (int): tolerance of prediction
            images([str]): list of relative paths to source images
        """
        self._y_line:int = y_line
        self._expected_width:int= expected_width
        self._tolerance:int = tolerance
        self.images = []
        for image in images:
            self.images.append(self.load_image(image))


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

    def preprocess(self, img_gray: np.ndarray) -> np.ndarray:
        """
        Local contrast normalization to make the gap signal consistent across
        different welding-arc glare and ambient lighting conditions. CLAHE is
        used over global hist-eq because the gap is a small low-contrast
        feature that global eq tends to wash out under bright glare.
        args:
            img_gray (np.ndarray): single-channel uint8 image
        returns:
            (np.ndarray): contrast-normalized uint8 image
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img_gray)

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

    def crop_to_weld_band(self, img_gray: np.ndarray, y_center: int, band_height: int = 5) -> np.ndarray:
        """
        Returns the column-wise mean of a small vertical band of rows around
        y_center as a single 1D row. Trades a few pixels of vertical
        resolution for a substantial SNR boost: per-pixel sensor / JPEG noise
        is uncorrelated across rows but the gap signal is correlated, so the
        mean averages the noise down while preserving the gap profile.
        args:
            img_gray(np.ndarray): greyscale image
            y_center(int): center row of the band
            band_height(int): total number of rows to average (odd is cleanest)
        returns:
            (np.ndarray): 1D uint8 row of width = img_gray.shape[1]
        """
        half = band_height // 2
        y0 = max(0, y_center - half)
        y1 = min(img_gray.shape[0], y_center + half + 1)
        band = img_gray[y0:y1, :].astype(np.float32)
        return np.mean(band, axis=0).astype(np.uint8)

    def pix_to_world(self):
        pass

    def _roi_bounds(self, image_shape: tuple, x_centre: int, roi_height: int = 100, roi_half_width: int = 200) -> tuple:
        """
        Compute the (x0, x1, y0, y1) bounds of the brief-required ROI crop.
        The ROI is a band centred vertically on y_line and horizontally on
        x_centre, clamped into the image. Used both for the
        Image{XXXX}_A_WeldGapPosition.JPG output and for the
        Image{XXXX}_B_InterimResultY.jpg interim outputs so all per-image
        artefacts share the same crop frame.
        args:
            image_shape: (H, W, ...) tuple
            x_centre: column the ROI is centred on (use predicted gap, or
                image-centre when the prediction is invalid).
            roi_height: total ROI height in pixels.
            roi_half_width: ROI half-width in pixels (so total width is 2x).
        returns:
            (x0, x1, y0, y1) — half-open bounds suitable for slicing
        """
        H, W = image_shape[0], image_shape[1]
        y0 = max(0, self._y_line - roi_height // 2)
        y1 = min(H, y0 + roi_height)
        x0 = max(0, int(x_centre) - roi_half_width)
        x1 = min(W, x0 + 2 * roi_half_width)
        x0 = max(0, x1 - 2 * roi_half_width)  #pull x0 in if we hit the right edge
        return x0, x1, y0, y1

    def _compute_interim_images(self, image: np.ndarray, position: int, valid: int,
                                 roi: tuple) -> list:
        """
        Returns a list of BGR np.ndarrays for the brief's
        Image{XXXX}_B_InterimResult{1..3}.jpg outputs. Default returns:
            1. raw colour ROI
            2. CLAHE-normalised greyscale ROI (the input most detectors
               actually consume)
            3. CLAHE'd ROI overlaid with the y_line guide and the
               predicted-x vertical line — quick visual check of where
               the detector landed relative to the brief's measurement row.
        Subclasses override to expose detector-specific intermediate
        results (e.g. matched-filter likelihood heatmap for SeamDP).
        args:
            image: full BGR image
            position: predicted x at y_line (or -1)
            valid: 1 = trust position, 0 = fail-safe
            roi: (x0, x1, y0, y1) crop bounds
        returns:
            list of np.ndarray BGR images (any number; only first 3 are saved).
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
        Brief-spec per-image artefact writer:
        - Image{XXXX}_A_WeldGapPosition.JPG : ROI crop with the determined
          weld-gap position drawn (cross at y_line if valid).
        - Image{XXXX}_B_InterimResult{1..3}.jpg : up to three detector-
          specific intermediate visualisations of the same ROI.
        ROI is centred horizontally on the prediction (or image-centre
        when invalid) so the marker is always visible inside the crop.
        """
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(fname)[0]
        W = image.shape[1]
        x_centre = position if (valid and position >= 0) else W // 2
        x0, x1, y0, y1 = self._roi_bounds(image.shape, x_centre)

        #A: ROI with the determined weld-gap position drawn. Per the brief
        # this is the FINAL detection result image — green cross when the
        # measurement is trusted, red when fail-safed (so a human inspecting
        # the InterimResults folder can immediately tell which frames the
        # closed-loop controller would actually act on).
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

        #B_InterimResult{1..3}: detector-specific intermediate ROIs. Brief
        # asks for "2-3 interim results"; anything beyond 3 is silently dropped.
        for i, interim in enumerate(self._compute_interim_images(image, position, valid, (x0, x1, y0, y1))[:3], start=1):
            cv2.imwrite(os.path.join(output_dir, f"{base}_B_InterimResult{i}.jpg"), interim)

    def process_dataset(self, image_dir: str, set_number, output_root: str = ".") -> str:
        """
        Brief-spec dataset processor. Iterates a Set folder in filename
        order and writes:
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

        args:
            image_dir: relative path to the Set folder (e.g. 'Set 1').
            set_number: 1, 2, or 3 — used in CSV filename and folder name.
            output_root: project root where InterimResultsOfSet<N>/ will go.
        returns:
            absolute path of the InterimResults folder that was written.
        """
        interim_dir = os.path.join(output_root, f"InterimResultsOfSet{set_number}")
        os.makedirs(interim_dir, exist_ok=True)

        files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(".jpg"))

        csv_path = os.path.join(interim_dir, f"PositionResultsOfSet{set_number}.csv")
        with open(csv_path, "w") as fh:
            #brief-mandated header — keep the spacing verbatim, the marking
            # script may parse columns by exact match.
            fh.write("ImageName, Weld gap position in pixel/integer , "
                     "Weld gap position valid? 0 = false, 1 = true\n")
            for fname in files:
                img = self.load_image(os.path.join(image_dir, fname))
                position, valid = self.process_image(img.copy())
                #fail-safe normalisation: brief mandates -1 / 0 for invalid
                if not valid or position is None or position < 0:
                    position, valid = -1, 0
                fh.write(f"{fname}, {int(position)}, {int(valid)}\n")
                self._save_outputs_for_image(img, fname, int(position), int(valid), interim_dir)

        return os.path.abspath(interim_dir)

    def draw_weld(self, image: np.ndarray, predicted_center: int, y_line: int = 70, valid: int = 1) -> np.ndarray:
        """
        Draws a cross at the predicted weld center.
        Green when the measurement passed validation, red when it was rejected
        (drawn at the rejected location for visualisation only — downstream
        code must treat valid=0 as no-measurement).
        args:
            image (np.ndarray): The image to draw on
            predicted_center (int): The predicted x coordinate of the weld center
            y_line (int): The y coordinate of the marker
            valid (int): 1 = passed validation, 0 = rejected
        returns:
            image (np.ndarray): The image with the cross drawn
        """
        if predicted_center is None or predicted_center < 0:
            return image
        color = (0, 255, 0) if valid else (0, 0, 255)
        cv2.drawMarker(image, (int(predicted_center), y_line), color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        return image
