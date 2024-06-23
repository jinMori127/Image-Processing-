# Ayal_Kaabia, 322784760
# Sami_Serhan, 327876298

import cv2
import numpy as np
import matplotlib.pyplot as plt


broken_img = cv2.imread('broken.jpg', cv2.IMREAD_GRAYSCALE)

# section a fix the broken img
bilateral_img = cv2.bilateralFilter(broken_img, d=3, sigmaColor=50, sigmaSpace=19)

median_img = cv2.medianBlur(bilateral_img, 5)

# Define the blurring mask G
G = np.array([[0, 1/6, 0],
              [1/6, 2/6, 1/6],
              [0, 1/6, 0]], dtype=np.float32)

B_blur = cv2.filter2D(median_img, -1, G)

B_sharp = cv2.subtract(median_img, B_blur)

lambda_factor = 0.5

# lamda * B_sharp
B_sharp_lamda = cv2.multiply(B_sharp, lambda_factor)

# B + lamda * B_sharp
B_enhanced = cv2.add(median_img, B_sharp_lamda)


plt.subplot(121)
plt.imshow(broken_img, cmap='gray')

plt.subplot(122)
plt.imshow(B_enhanced, cmap='gray')
plt.show()

# save the img
path = 'fixed_broken.jpg'
cv2.imwrite(path, B_enhanced)

# section B law of large numbers
# noise images
noised_images = np.load('noised_images.npy')

mean_img = np.mean(noised_images, axis=0)

median_img = np.median(noised_images, axis=0)

path = 'mean_fixed_noised_images_.jpg'
cv2.imwrite(path, mean_img)

path = 'median_fixed_noised_images_.jpg'
cv2.imwrite(path, median_img)

plt.subplot(121)
plt.imshow(mean_img, cmap='gray')

plt.subplot(122)
plt.imshow(median_img, cmap='gray')
plt.show()

