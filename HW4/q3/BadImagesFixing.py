# Ayal_Kaabia, 322784760
# Sami_Serhan, 327876298

# Please replace the above comments with your names and ID numbers in the same format.

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt


baby_img1 = np.float32([[6,19],[111,19],[111,131],[6,131]])   
baby_img2 = np.float32([[181,4],[250,70],[177,121],[120,50]]) 
baby_img3 = np.float32([[77,162],[146,116],[246,160],[132,245]]) 



def transform_img(target_img, src, dist = np.float32([[0,0],[256,0],[256,256],[0,256]])):
    T = cv2.getPerspectiveTransform(src, dist)
    target_img_transformed = cv2.warpPerspective(target_img, T, (256,256), flags=cv2.INTER_LINEAR)

    print(np.std(target_img_transformed))

    bilateral_img = cv2.bilateralFilter(target_img_transformed, d=3, sigmaColor=40, sigmaSpace=19)

    img_median = cv2.medianBlur(bilateral_img,7)

    plt.imshow(img_median, cmap='gray', vmin=0, vmax=255)
    plt.show()
    return img_median
    


def clean_baby(im):
	# Your code goes here
    img1_clean_transformed  = transform_img(im,baby_img1)
    img2_clean_transformed  = transform_img(im,baby_img2)
    img3_clean_transformed  = transform_img(im,baby_img3)
    
    noised_images = np.zeros((3, 256, 256))
    noised_images[0, :, :] = img1_clean_transformed
    noised_images[1, :, :] = img2_clean_transformed
    noised_images[2, :, :] = img3_clean_transformed

    median_img = np.median(noised_images, axis=0)

    plt.imshow(median_img, cmap='gray', vmin=0, vmax=255)
    plt.show()
    return median_img
# first we will choose the point to transform the images to 


def clean_windmill(im):
    # Step 1:Compute Fourier transform of the original image
    fourier_transform = fft2(im)
    # Step 2: Shift the Fourier transform to the middle
    shifted_transform = fftshift(fourier_transform)
    # Step 3: Find the highest symmetric peaks in the shifted Fourier transform(using paint)
    # Step 4: Weaken the identified peaks or zero them
    shifted_transform[124][100]=0
    shifted_transform[132][156]=0
    # Step 5: Perform inverse Fourier transform
    inverse_transform = ifftshift(shifted_transform)
    cleaned_image = np.abs(ifft2(shifted_transform))

    return cleaned_image

def clean_watermelon(im):
    kernel = np.array([[0, -7, 0],
                       [-7, 28, -7],
                       [0, -7, 0]], dtype=np.float32)

    # Apply the high-pass filter (convolution)
    im_filtered = cv2.filter2D(im, -1, kernel)

    # Add the filtered image to the original to sharpen it
    im_clean = cv2.add(im, im_filtered)

    return im_clean

def clean_umbrella(im):
    # extract u and v 
    v = np.fft.fftfreq(im.shape[0])
    u = np.fft.fftfreq(im.shape[1])
    [X, Y] = np.meshgrid(u, v)

    x0, y0 = 79, 4  

    # calculate Fourier for moved delta by 1 + e^(-2j * pi * (u * x0 + v * y0))
    fourier_delta = 1 + np.exp(-2j * np.pi * (X * x0 + Y * y0))
    # make sure that we will not devide by zeros 
    fourier_delta[np.abs(fourier_delta) < 0.001] = 1

    # calculate the Fourier for the original image and divide by modifier

    fourier_original = 2* (np.fft.fft2(im) / (fourier_delta))

    # calc the inverse Fourier transform for fourier_original to get the original img
    original_img =  np.abs(np.fft.ifft2(fourier_original))
    return original_img

def clean_USAflag(im):
    clean_im = im.copy()
    stars_side = im[0:91, 0:179]
    no_change = im[92:168, 0:71]
    r = 6
    for i in range(im.shape[0]):
        for j in range(r, im.shape[1]-r):
            clean_im[i][j] = np.median(im[i, j - r : j + r + 1])    

    clean_im[0:91, 0:179] = stars_side
    clean_im[92:168, 0:71] = no_change

    return clean_im

def clean_house(im):

    img_fourier=np.fft.fft2(im)
    # kernel for horizontal motion blur
    blur_mask = np.zeros(im.shape)

    
    blur_mask[0,0:10] = 0.1

    blur_mask = np.fft.fft2(blur_mask)
    blur_mask[np.abs(blur_mask) < 0.001] = 1

    img_fourier = img_fourier/blur_mask
    cleaned_img = abs(np.fft.ifft2(img_fourier))
    return cleaned_img


def clean_bears(im):
    # stretch the contrast to the full range of 0 - 255 
    normalized_image = cv2.normalize(im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return normalized_image


