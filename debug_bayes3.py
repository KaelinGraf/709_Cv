from weld_detect import WeldDetector
import numpy as np
import cv2

weld_detector = WeldDetector(images=['WeldGapImages_export/Set 1/image0015.jpg'])
image = weld_detector.images[0]
image_greyscale = weld_detector.rgb_to_greyscale(image)
weld_line_slice = weld_detector.crop_to_weld_line(image_greyscale, 70)

pdfs = []
for template in weld_detector.templates:
    res = cv2.matchTemplate(weld_line_slice, template, cv2.TM_CCOEFF_NORMED).flatten()
    pdf = np.maximum(0, res) ** 2
    if np.sum(pdf) > 0:
        pdf /= np.sum(pdf)
        pdfs.append(pdf)

pdfs = np.array(pdfs)
print("Naive Bayes with small epsilon (1e-10):")
epsilon = 1e-10
summed_logs = np.sum(np.log(pdfs + epsilon), axis=0)
comb = np.exp(summed_logs) / np.sum(np.exp(summed_logs))
print("Max at:", np.argmax(comb), "EV:", np.sum(np.arange(len(comb)) * comb))

print("Naive Bayes with generous base probability (mean of uniform dist):")
epsilon = 1.0 / len(pdfs[0])
summed_logs = np.sum(np.log(pdfs + epsilon), axis=0)
comb = np.exp(summed_logs) / np.sum(np.exp(summed_logs))
print("Max at:", np.argmax(comb), "EV:", np.sum(np.arange(len(comb)) * comb))

print("Linear Opinion Pool (Arithmetic Mean):")
comb = np.mean(pdfs, axis=0)
comb /= np.sum(comb)
print("Max at:", np.argmax(comb), "EV:", np.sum(np.arange(len(comb)) * comb))

