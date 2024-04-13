import cv2

img = cv2.imread('1.png', 1)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray_img)
cv2.waitKey(0)

new = cv2.equalizeHist(gray_img)
cv2.imshow('new', new)
cv2.waitKey(0)
