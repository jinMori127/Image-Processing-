# Ayal_Kaabia, 322784760
# Sami_Serhan, 327876298

import cv2
import numpy as np
import matplotlib.pyplot as plt

def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
    # making sure that we are working with float64
    f_img = im.astype(np.float64)

    res_img = np.zeros_like(f_img)
    rows, cols = im.shape

    print(np.std(im))

    # define the function gs
    x, y = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1))
    gs = np.exp(-(x**2 + y**2) / (2 * stdSpatial**2))

    for i in range(rows):
        for j in range(cols):
            # defining the boundaries of the window around i,j pixel
            row_min, row_max = max(i - radius, 0), min(i + radius + 1, rows)
            col_min, col_max = max(j - radius, 0), min(j + radius + 1, cols)

            # define the window
            window = f_img[row_min:row_max, col_min:col_max]

            # calculate gi/gs for each pixel in the window since we have no fixes window size
            gs_window = gs[row_min-i+radius:row_max-i+radius, col_min-j+radius:col_max-j+radius]

            gi = np.exp(-((window - f_img[i, j])**2) / (2 * stdIntensity**2))

            Wp = gs_window * gi
            res_img[i, j] = np.sum(Wp * window) / np.sum(Wp)

    return np.clip(res_img, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    # change this to the name of the image you'll try to clean up
    img_path_list = ['balls.jpg', 'taj.jpg', 'NoisyGrayImage.png']
    list_spatial = [6.47, 8.49, 6.87]
    list_intensity = [20, 30, 100]
    list_radius = [3, 9, 9]
    for i in range(3):
        image = cv2.imread(img_path_list[i], cv2.IMREAD_GRAYSCALE)

        clear_image_b = clean_Gaussian_noise_bilateral(image, list_radius[i], list_spatial[i], list_intensity[i])

        sol_file = f'cleaned_{img_path_list[i]}'
        cv2.imwrite(sol_file, clear_image_b)
        plt.subplot(121)
        plt.imshow(image, cmap='gray')

        plt.subplot(122)
        plt.imshow(clear_image_b, cmap='gray')
        plt.show()
