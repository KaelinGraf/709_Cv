"""
Ground-truth labelling tool for weld-gap images.

Run from the project root:
    python -m utils.label_ground_truth --set "Set 1"

For each image it shows a zoomed crop centred on the mean of the existing
detectors' predictions, draws the y=70 line, and overlays each detector's
prediction. Click on the gap centre to record the ground-truth x position
in the ORIGINAL (un-zoomed) image coordinate system; press 'n' to mark
"no visible gap / unsure" (saved as -1); 'b' to back up one image; 'q' to
quit (saves progress). Labels are written incrementally to
ground_truth/<set>.json so you can resume.

Output JSON format:
    { "image0001.jpg": 975, "image0002.jpg": -1, ... }
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

#Allow `python utils/label_ground_truth.py` invocation as well.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from detectors import (
    WeldDetectorTemplateMatching,
    WeldDetectorCanny,
    WeldDetectorSeamDP,
)


Y_LINE = 70                  #brief-defined ground-truth row
CROP_HEIGHT = 80             #band of rows shown in the zoomed view
CROP_HALF_WIDTH = 150        #+- columns around the centre estimate (wider to
                             # absorb detector disagreement without cutting off
                             # the true gap)
ZOOM = 4                     #display magnification for the crop
DISPLAY_FULL_HEIGHT = 600    #resize the full image to this height for context
LOWER_REFERENCE_Y = 180      #y-row of the secondary reference zoom — for
                             # Set 3-style frames the gap is occluded at
                             # y_line by glare but visible at this y; showing
                             # both lets the user verify extrapolation


def _load_existing(path):
    if os.path.exists(path):
        with open(path, "r") as fh:
            return json.load(fh)
    return {}


def _save(labels, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(labels, fh, indent=2, sort_keys=True)


def _draw_predictions(img, predictions, y_line):
    """Draw each detector's prediction on a copy of img and return it."""
    out = img.copy()
    cv2.line(out, (0, y_line), (out.shape[1] - 1, y_line), (0, 255, 255), 1)
    palette = {
        "TM": (0, 255, 0),       #green
        "Canny": (255, 0, 0),    #blue
        "SeamDP": (0, 165, 255), #orange
    }
    for name, (pos, valid) in predictions.items():
        if pos is None or pos < 0 or not valid:
            continue
        cv2.drawMarker(out, (int(pos), y_line), palette[name],
                       markerType=cv2.MARKER_TRIANGLE_DOWN, markerSize=14, thickness=1)
    return out


def _make_zoom(img, centre_x, y_line):
    """
    Crop a CROP_HEIGHT-tall band centred on y_line and CROP_HALF_WIDTH on
    each side of centre_x, then upscale by ZOOM with nearest-neighbour
    interpolation so individual pixels are clickable.

    Returns (zoom_img, x0, y0) where (x0, y0) is the top-left corner of
    the crop in original-image coordinates.
    """
    H, W = img.shape[:2]
    x0 = max(0, int(centre_x) - CROP_HALF_WIDTH)
    x1 = min(W, x0 + 2 * CROP_HALF_WIDTH)
    x0 = max(0, x1 - 2 * CROP_HALF_WIDTH)  #ensure full-width crop where possible
    y0 = max(0, y_line - CROP_HEIGHT // 2)
    y1 = min(H, y0 + CROP_HEIGHT)
    y0 = max(0, y1 - CROP_HEIGHT)
    crop = img[y0:y1, x0:x1]
    zoom = cv2.resize(crop, (crop.shape[1] * ZOOM, crop.shape[0] * ZOOM),
                      interpolation=cv2.INTER_NEAREST)
    return zoom, x0, y0


class Labeler:
    def __init__(self, img_dir, output_path, only=None, force=False):
        """
        args:
            only (list[str] or None): if given, only step through these
                filenames (in the order provided). Use to revisit specific
                images for re-review without scrolling through the whole set.
            force (bool): if True, present every image regardless of whether
                it already has a label. Combined with --only this is
                useful for re-checking suspect labels.
        """
        self.img_dir = img_dir
        self.output_path = output_path
        self.labels = _load_existing(output_path)
        self.force = force
        if only:
            self.files = list(only)
        else:
            #freeze a sorted file list so 'b' (back up) is deterministic
            self.files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".jpg"))
        #shared detectors — instantiated once, no images bound (we feed
        # them per-frame). Loaded width=2080 is read from the first image.
        sample = cv2.imread(os.path.join(img_dir, self.files[0]))
        W = sample.shape[1]
        sxr = (int(W * 0.2), int(W * 0.8))
        self.tm = WeldDetectorTemplateMatching(images=[], search_x_range=sxr)
        self.canny = WeldDetectorCanny(images=[], search_x_range=sxr)
        self.seam = WeldDetectorSeamDP(images=[], search_x_range=sxr)

        self._click_x = None  #set by mouse callback in original-image coords

    def _on_click_zoom(self, event, x, y, flags, param):
        """Click in the zoomed-crop window — map back to original-image x."""
        if event == cv2.EVENT_LBUTTONDOWN:
            x0, y0 = param
            self._click_x = int(x0 + x // ZOOM)

    def _on_click_full(self, event, x, y, flags, param):
        """
        Click in the full-image preview window — map display-pixel x back
        to original-image x via the resize scale. Lets the user label
        when the zoomed crop happens to miss the true gap (e.g. when
        detectors disagree wildly, the crop centres on the wrong place).
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            scale = param
            self._click_x = int(round(x / scale))

    def _predictions(self, img):
        """Run all three detectors on a fresh copy of the image."""
        return {
            "TM": self.tm.process_image(img.copy()),
            "Canny": self.canny.process_image(img.copy()),
            "SeamDP": self.seam.process_image(img.copy()),
        }

    def run(self):
        idx = 0
        while idx < len(self.files):
            fname = self.files[idx]
            if fname in self.labels and not self.force:
                idx += 1
                continue

            path = os.path.join(self.img_dir, fname)
            img = cv2.imread(path)
            if img is None:
                print(f"  [skip] cannot read {path}")
                idx += 1
                continue

            preds = self._predictions(img)
            #centre the zoom on SeamDP's prediction when valid (most reliable
            # detector on this dataset), otherwise the mean of valid detector
            # predictions, otherwise image-centre. Avoids the case where TM /
            # Canny disagree wildly and pull the crop centre off the true gap.
            seam_pos, seam_valid = preds.get("SeamDP", (-1, 0))
            if seam_valid and seam_pos >= 0:
                centre = int(seam_pos)
            else:
                valid_x = [p for (p, v) in preds.values() if v == 1 and p >= 0]
                centre = int(np.mean(valid_x)) if valid_x else img.shape[1] // 2

            annotated = _draw_predictions(img, preds, Y_LINE)
            #full-image preview, scaled to a manageable height. Also clickable
            # so the user can label when the zoomed crop misses the gap.
            # We pre-create both windows by name so setMouseCallback always
            # has a valid handler, and avoid special characters in the names
            # (Qt's window-name parsing trips on em-dashes etc).
            scale = DISPLAY_FULL_HEIGHT / annotated.shape[0]
            preview = cv2.resize(annotated, (int(annotated.shape[1] * scale), DISPLAY_FULL_HEIGHT))
            cv2.namedWindow("full preview (clickable)", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("full preview (clickable)", preview)
            cv2.setMouseCallback("full preview (clickable)", self._on_click_full, param=scale)

            zoom, x0, y0 = _make_zoom(annotated, centre, Y_LINE)
            zoom_y_line = (Y_LINE - y0) * ZOOM + ZOOM // 2
            cv2.line(zoom, (0, zoom_y_line), (zoom.shape[1] - 1, zoom_y_line), (0, 255, 255), 1)
            cv2.namedWindow("click gap centre", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("click gap centre", zoom)
            cv2.setMouseCallback("click gap centre", self._on_click_zoom, param=(x0, y0))

            #Secondary reference zoom — same x-range but at LOWER_REFERENCE_Y.
            # When the gap is glare-occluded at y=70 (e.g. Set 3) it is
            # typically still visible here; the user can see the gap below
            # the glare and mentally project it back up to y=70 for a
            # sanity-check on the SeamDP extrapolation.
            if annotated.shape[0] > LOWER_REFERENCE_Y + CROP_HEIGHT:
                ref_zoom, ref_x0, ref_y0 = _make_zoom(annotated, centre, LOWER_REFERENCE_Y)
                ref_y_line = (LOWER_REFERENCE_Y - ref_y0) * ZOOM + ZOOM // 2
                cv2.line(ref_zoom, (0, ref_y_line), (ref_zoom.shape[1] - 1, ref_y_line),
                         (255, 0, 255), 1)
                cv2.namedWindow("reference zoom (visible gap)", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("reference zoom (visible gap)", ref_zoom)

            preds_str = "  ".join(f"{n}={p}/{v}" for n, (p, v) in preds.items())
            existing = self.labels.get(fname)
            existing_str = f"  current label: {existing}" if existing is not None else ""
            print(f"\n[{idx + 1}/{len(self.files)}] {fname}   detectors: {preds_str}{existing_str}")
            print("  click gap | n=no-gap | b=back | s=skip | q=quit")

            self._click_x = None
            while True:
                key = cv2.waitKey(20) & 0xFF
                if self._click_x is not None:
                    self.labels[fname] = self._click_x
                    print(f"  recorded x={self._click_x}")
                    _save(self.labels, self.output_path)
                    idx += 1
                    break
                if key == ord("n"):
                    self.labels[fname] = -1
                    print("  recorded NO-GAP (-1)")
                    _save(self.labels, self.output_path)
                    idx += 1
                    break
                if key == ord("s"):
                    print("  skipped (no label saved)")
                    idx += 1
                    break
                if key == ord("b"):
                    if idx > 0:
                        idx -= 1
                        prev = self.files[idx]
                        if prev in self.labels:
                            del self.labels[prev]
                            _save(self.labels, self.output_path)
                            print(f"  cleared label for {prev}")
                    break
                if key == ord("q"):
                    print("  quitting")
                    cv2.destroyAllWindows()
                    return

        cv2.destroyAllWindows()
        print(f"\nDone. {len(self.labels)} labels saved to {self.output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="Set 1",
                   help='Subfolder of WeldGapImages_export to label, e.g. "Set 1"')
    p.add_argument("--root", default="WeldGapImages_export",
                   help="Image root directory (default: WeldGapImages_export)")
    p.add_argument("--output_dir", default="ground_truth",
                   help="Directory to write the labels JSON")
    p.add_argument("--only", nargs="+", default=None,
                   help="Step through only these filenames (re-review mode).")
    p.add_argument("--force", action="store_true",
                   help="Re-prompt even for already-labelled images.")
    args = p.parse_args()

    img_dir = os.path.join(args.root, args.set)
    if not os.path.isdir(img_dir):
        raise SystemExit(f"image dir not found: {img_dir}")
    safe = args.set.replace(" ", "_").lower()
    output_path = os.path.join(args.output_dir, f"{safe}.json")

    Labeler(img_dir, output_path, only=args.only, force=args.force).run()


if __name__ == "__main__":
    main()
