import numpy as np
import cv2
import matplotlib.pyplot as plt


name = "./G30.jpg"

# image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
# cv2.imshow("orginal", image)
# shape = image.shape
# mean = 10
# var = 50
# sigma = var**0.5
# gauss = np.random.normal(mean,sigma,(shape))
# noisy = image + gauss/2
# print(image)
# print(gauss)
# cv2.imshow("GB", noisy)


img = cv2.imread(name)
cv2.imshow("orginal", img)
img = cv2.GaussianBlur(img,(23, 23),0)
cv2.imshow("orginalGB", img)
cv2.imwrite("./GB_23_15_25.png", img)
img = cv2.Canny(img, 15, 25, L2gradient=True)
cv2.imshow("canny", img)
cv2.imwrite("./GB_23_15_25.jpg", img)


# img2 = cv2.imread(name)
# # cv2.imshow("orginal1", img2)
# img2 = cv2.medianBlur(img2, 17)
# cv2.imshow("orginalMB", img2)
# # cv2.imwrite("./MB_17_35_50.png", img2)
# img2 = cv2.Canny(img2, 35, 50, L2gradient=True)
# cv2.imshow("canny1", img2)
# # cv2.imwrite("./MB_17_35_50.jpg", img2)
cv2.waitKey(0)