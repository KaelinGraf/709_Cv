from weld_detect import WeldDetector
import numpy as np
import cv2

weld_detector = WeldDetector(images=['WeldGapImages_export/Set 1/image0015.jpg'])
image = weld_detector.images[0]
image_greyscale = weld_detector.rgb_to_greyscale(image)
weld_line_slice = weld_detector.crop_to_weld_line(image_greyscale, 70)

res_list = []
for template in weld_detector.templates:
    res = cv2.matchTemplate(weld_line_slice, template, cv2.TM_CCOEFF_NORMED).flatten()
    res_list.append(res)

res_arr = np.array(res_list) # shape (11, len)
print("Res at 940:")
print(res_arr[:, 940])
print("Res at 967:")
print(res_arr[:, 967])
print("Res at 1430:")
print(res_arr[:, 1430])

