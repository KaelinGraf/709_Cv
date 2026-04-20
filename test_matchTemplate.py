import cv2
import numpy as np
a = np.random.rand(100).astype(np.float32)
template = np.random.rand(10).astype(np.float32)
res = cv2.matchTemplate(a, template, cv2.TM_CCOEFF_NORMED)
print(res.shape)
