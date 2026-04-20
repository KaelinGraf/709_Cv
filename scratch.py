import numpy as np
import cv2

with open("weld_detect.py", "r") as f:
    text = f.read()

new_text = text.replace(
    "return int(np.argmax(final_pdf))",
    "final_pdf_blurred = np.convolve(final_pdf, np.ones(50)/50, mode='same')\n        print('blurred argmax:', np.argmax(final_pdf_blurred))\n        return int(np.argmax(final_pdf_blurred))"
)

with open("weld_detect_test.py", "w") as f:
    f.write(new_text)

