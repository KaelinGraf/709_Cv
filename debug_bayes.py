from weld_detect import WeldDetector
import numpy as np
import cv2

weld_detector = WeldDetector(images=['WeldGapImages_export/Set 1/image0015.jpg'])
image = weld_detector.images[0]
image_greyscale = weld_detector.rgb_to_greyscale(image)
weld_line_slice = weld_detector.crop_to_weld_line(image_greyscale, 70)

pdfs = []
expected_values = []
for template in weld_detector.templates:
    res = cv2.matchTemplate(weld_line_slice, template, cv2.TM_CCOEFF_NORMED).flatten()
    res_rectified = np.maximum(0, res)
    res_filtered = res_rectified ** 2
    area = np.sum(res_filtered)
    if area > 0:
        pdf = res_filtered / area
        x_indices = np.arange(len(pdf))
        mu = np.sum(x_indices * pdf)
        # Pad pdf to same length (image width) if necessary to avoid object array issues
        pad_width = weld_line_slice.shape[0] - len(pdf)
        pdf = np.pad(pdf, (0, pad_width), 'constant')
        pdfs.append(pdf)
        expected_values.append(mu)

pdfs = np.array(pdfs)
print("Expected values mapping to each template:", expected_values)
print("Max probabilities at:", np.argmax(pdfs, axis=1))

epsilon = 1e-10
summed_logs = np.sum(np.log(pdfs + epsilon), axis=0)
unnormalized_combined = np.exp(summed_logs)
final_pdf = unnormalized_combined / np.sum(unnormalized_combined)
final_mu = np.sum(np.arange(len(final_pdf)) * final_pdf)
print("Final expected value:", final_mu)
print("Final max prob at:", np.argmax(final_pdf))
