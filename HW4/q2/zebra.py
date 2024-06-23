# Ayal_Kaabia, 322784760
# Sami_Serhan, 327876298

# Please replace the above comments with your names and ID numbers in the same format.

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt

image_path = 'zebra.jpg'
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Your code goes here

#DOUBLE SIZED CODE:
# Compute Fourier transform of the original image
fourier_transform = fft2(gray_image)
shifted_transform = fftshift(fourier_transform)
magnitude_spectrum = np.log(1 + np.abs(shifted_transform))

# Zero padding the Fourier transform
pad_height = shifted_transform.shape[0] * 2
pad_width = shifted_transform.shape[1] * 2

# Create an array of zeros with the doubled size
padded_transform = np.zeros((pad_height, pad_width), dtype=np.complex128)

# Compute the starting indices to insert the original Fourier transform in the center
start_row = (pad_height - shifted_transform.shape[0]) // 2
start_col = (pad_width - shifted_transform.shape[1]) // 2

# Insert the original Fourier transform into the center of the zero-padded array
padded_transform[start_row:start_row + shifted_transform.shape[0],
                  start_col:start_col + shifted_transform.shape[1]] = shifted_transform


# Compute inverse Fourier transform of the padded transform and multiply by 4 to brighten
restored_image = np.abs(ifft2(ifftshift(padded_transform)))
restored_image*=4
doubled_image=restored_image.astype(np.uint8)
double_magnitude_spectrum = np.log(1 + np.abs(padded_transform))

#CODE FOR 4 IMAGES
# Compute Fourier transform of the original image
four_fourier_transform = fft2(gray_image)
four_shifted_transform = fftshift(four_fourier_transform)

# Get the shape of the Fourier transform
M, N = four_shifted_transform.shape

# Create an array for the padded Fourier transform
four_padded_transform = np.zeros((2 * M - 1, 2 * N - 1), dtype=np.complex128)

# Insert the Fourier coefficients with zero padding
for i in range(M):
    for j in range(N):
        four_padded_transform[2*i, 2*j] = four_shifted_transform[i, j]

# Compute inverse Fourier transform of the padded transform
four_image = np.abs(ifft2(ifftshift(four_padded_transform)))
four_image*=4
four_image=four_image.astype(np.uint8)
# Compute Fourier transform magniute of the four image
four_magnitude_spectrum = np.log(1 + np.abs(four_padded_transform))



plt.figure(figsize=(10,10))
plt.subplot(321)
plt.title('Original Grayscale Image')
plt.imshow(gray_image, cmap='gray')

plt.subplot(322)
plt.title('Fourier Spectrum')
plt.imshow(magnitude_spectrum, cmap='gray')

plt.subplot(323)
plt.title('Fourier Spectrum Zero Padding')
plt.imshow(double_magnitude_spectrum, cmap='gray')

plt.subplot(324)
plt.title('Two Times Larger Grayscale Image')
plt.imshow(doubled_image, cmap='gray')

plt.subplot(325)
plt.title('Fourier Spectrum Four Copies')
plt.imshow(four_magnitude_spectrum, cmap='gray')

plt.subplot(326)
plt.title('Four Copies Grayscale Image')
plt.imshow(four_image, cmap='gray')

# Adjust layout to prevent overlapping
plt.tight_layout()

plt.savefig('zebra_scaled.png')