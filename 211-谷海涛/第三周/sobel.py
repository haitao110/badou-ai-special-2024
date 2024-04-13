import cv2
import numpy as np

img = cv2.imread('1.png')
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 1, 0)

absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow('dst', dst)
cv2.waitKey(0)