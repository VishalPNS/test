import cv2
import matplotlib.pyplot as plt
import numpy as np

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]

len(flags)

flags[20]

nemo = cv2.imread('face.jpg')
plt.imshow(nemo)
plt.show()