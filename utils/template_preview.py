import cv2
import os
import numpy as np

cwd = os.getcwd()

files_list = [file for file in os.listdir(cwd) if "template" in file and ".npy" in file  ]



for file in files_list:
    signal = np.load(file)
    print(signal)