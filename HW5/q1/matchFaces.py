# Sami Serhan, 327876298
# Ayal Kaabia, 322784760

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift

import warnings
warnings.filterwarnings("ignore")


def scale_down(image, resize_ratio):
    height, width = image.shape[:2]

    image = cv2.GaussianBlur(image, (5, 5), 0)

    # calculate the Fourier transform
    f_im = fft2(image)
    fshifted_im = fftshift(f_im)

    # calculate the new image size
    new_scaled_height, new_scaled_width = int(height * resize_ratio), int(width * resize_ratio)

    # calculate the where to crop the frequency
    from_width, to_width = (width - new_scaled_width) // 2, (width + new_scaled_width) // 2
    from_height, to_height = (height - new_scaled_height) // 2, (height + new_scaled_height) // 2

    fshifted_im = fshifted_im[from_height:to_height, from_width:to_width]

    # calculate the inverse Fourier Transform
    fshifted_im = ifftshift(fshifted_im)
    resized_image = np.abs(ifft2(fshifted_im))

    f_resized_image = np.uint8(cv2.normalize(resized_image, None, 0, 255, cv2.NORM_MINMAX))
    return f_resized_image


def scale_up(image, resize_ratio):
    height, width = image.shape[:2]

    f_im = fft2(image)
    fshifted_im = fftshift(f_im)

    new_scaled_height, new_scaled_width = int(height * resize_ratio), int(width * resize_ratio)
    new_scaled_fourier = np.zeros((new_scaled_height, new_scaled_width), dtype=fshifted_im.dtype)

    # calculate the where to we will put the new image frequencies
    from_height, from_width = (new_scaled_height - height) // 2, (new_scaled_width - width) // 2

    # put the original frequency of the image in the new scaled image frequency
    new_scaled_fourier[from_height:from_height+height, from_width:from_width+width] = fshifted_im

    # calculate the inverse Fourier Transform
    fshift_im = np.fft.ifftshift(new_scaled_fourier)
    resized_image = np.abs(np.fft.ifft2(fshift_im))
    f_resized_image = np.uint8(cv2.normalize(resized_image, None, 0, 255, cv2.NORM_MINMAX))
    return f_resized_image


def ncc_2d(image, pattern):
    windows = np.lib.stride_tricks.sliding_window_view(image, pattern.shape)

    # calculate the mean for each window and for the pattern
    windows_mean = np.mean(windows, axis=(2, 3), keepdims=True)
    pattern_mean = np.mean(pattern)

    # we will calculate the given:
    # sum((windows - windows_mean)*(pattern-pattern_mean)) / sqrt(sum((windows-windows_mean)^2) * sum((pattern-pattern_mean)^2))
    numerator = np.sum((windows - windows_mean)*(pattern - pattern_mean), axis=(2, 3))
    denominator = np.sqrt(np.sum((windows - windows_mean)**2, axis=(2, 3)) * np.sum((pattern-pattern_mean)**2))

    # case that we get 0 in the denominator will set it to one and then set the numerator to 0
    # so when we divide 0/1 = 0 and that exactly what we want
    denominator[denominator == 0] = 1
    numerator[denominator == 0] = 0

    ncc_2d = numerator / denominator
    return ncc_2d



def display(image, pattern):

    plt.subplot(2, 3, 1)
    plt.title('Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title('Pattern')
    plt.imshow(pattern, cmap='gray', aspect='equal')

    ncc = ncc_2d(image, pattern)

    plt.subplot(2, 3, 5)
    plt.title('Normalized Cross-Correlation Heatmap')
    plt.imshow(ncc ** 2, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')

    cbar = plt.colorbar()
    cbar.set_label('NCC Values')

    plt.show()


def draw_matches(image, matches, pattern_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for point in matches:
        y, x = point
        top_left = (int(x - pattern_size[1]/2), int(y - pattern_size[0]/2))
        bottom_right = (int(x + pattern_size[1]/2), int(y + pattern_size[0]/2))
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 1)

    plt.imshow(image, cmap='gray')
    plt.show()

    cv2.imwrite(f"{CURR_IMAGE}_result.jpg", image)


# ############# Students #############
CURR_IMAGE = "students"

students_img = cv2.imread(f'{CURR_IMAGE}.jpg')
students_img = cv2.cvtColor(students_img, cv2.COLOR_BGR2GRAY)

pattern = cv2.imread('template.jpg')

pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)
image_scaled = scale_up(students_img, resize_ratio=1.32)
pattern_scaled = scale_down(pattern, resize_ratio=0.75)

display(image_scaled, pattern_scaled)

ncc = ncc_2d(image_scaled, pattern_scaled)
threshold = 0.51
real_matches_indices = np.where(ncc >= threshold)
real_matches = np.column_stack(real_matches_indices)

######### DONT CHANGE THE NEXT TWO LINES #########
real_matches[:, 0] += pattern_scaled.shape[0] // 2			# if pattern was not scaled, replace this with "pattern"
real_matches[:, 1] += pattern_scaled.shape[1] // 2			# if pattern was not scaled, replace this with "pattern"

# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.
resize_ratio_inverse = 1 / 1.32
real_matches = real_matches * resize_ratio_inverse

draw_matches(students_img, real_matches, pattern_scaled.shape)  # if pattern was not scaled, replace this with "pattern"

#
#
#
#
# ############# Crew #############
CURR_IMAGE = "thecrew"
#
crew_image = cv2.imread(f'{CURR_IMAGE}.jpg')
crew_image = cv2.cvtColor(crew_image, cv2.COLOR_BGR2GRAY)
#
pattern = cv2.imread('template.jpg')
#
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

image_scaled = scale_up(crew_image, resize_ratio=2.52)
pattern_scaled = scale_down(pattern, resize_ratio=0.58)

display(image_scaled, pattern_scaled)

ncc = ncc_2d(image_scaled, pattern_scaled)
threshold = 0.441
real_matches_indices = np.where(ncc >= threshold)
real_matches = np.column_stack(real_matches_indices)

######### DONT CHANGE THE NEXT TWO LINES #########
real_matches[:, 0] += pattern_scaled.shape[0] // 2			# if pattern was not scaled, replace this with "pattern"
real_matches[:, 1] += pattern_scaled.shape[1] // 2			# if pattern was not scaled, replace this with "pattern"

# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.
crew_resize_ratio_inverse = 1 / 2.52
real_matches = real_matches * crew_resize_ratio_inverse

draw_matches(crew_image, real_matches, pattern_scaled.shape) # if pattern was not scaled, replace this with "pattern"
