
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import warnings

warnings.filterwarnings("ignore")


# Input: numpy array of images and number of gray levels to quantize the images down to
# Output: numpy array of images, each with only n_colors gray levels
def quantization(imgs_arr, n_colors=4):
    img_size = imgs_arr[0].shape
    res = []

    for img in imgs_arr:
        X = img.reshape(img_size[0] * img_size[1], 1)
        km = KMeans(n_clusters=n_colors)
        km.fit(X)

        img_compressed = km.cluster_centers_[km.labels_]
        img_compressed = np.clip(img_compressed.astype('uint8'), 0, 255)

        res.append(img_compressed.reshape(img_size[0], img_size[1]))
    return np.array(res)


# Input: A path to a folder and formats of images to read
# Output: numpy array of grayscale versions of images read from input folder, and also a list of their names
def read_dir(folder, formats=(".jpg", ".png")):
    image_arrays = []
    lst = [file for file in os.listdir(folder) if file.endswith(formats)]
    for filename in lst:
        file_path = os.path.join(folder, filename)
        image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_arrays.append(gray_image)
    return np.array(image_arrays), lst


# Input: image object (as numpy array) and the index of the wanted bin (between 0 to 9)
# Output: the height of the idx bin in pixels
def get_bar_height(image, idx):
    # Assuming the image is of the same pixel proportions as images supplied in this exercise, the following values will work
    x_pos = 70 + 40 * idx
    y_pos = 274
    # the bars color is black(value=0)
    while image[y_pos, x_pos] == 0:
        y_pos -= 1
    return 274 - y_pos


# Sections c, d
# Remember to uncomment compare_hist before using it!

def compare_hist(src_image, target):
    target_hist = cv2.calcHist([target], [0], None, [256], [0, 256]).flatten()
    target_cumulative_hist = np.cumsum(target_hist)
    # making windows of size height 15 and width 10 as the size of the numbers pictures
    windows = np.lib.stride_tricks.sliding_window_view(src_image, target.shape)
    # this is the target window that the top number should be in
    # (note found this specific window by plt.grid, and also
    # I checked the top left windows and print the wanted window indexes 113,30)

    image_expected_window = windows[113, 30]
    window_hist = cv2.calcHist([image_expected_window], [0], None, [256], [0, 256]).flatten()
    cumulative_hist = np.cumsum(window_hist)
    # calculate the emd distance for the target and our checked window
    emd_distance = np.sum(np.abs(cumulative_hist - target_cumulative_hist))

    if emd_distance < 260:
        return True

    return False


images, names = read_dir('data')
numbers, _ = read_dir('numbers')
max_student_num = []

# moving over all the images
for img_idx in range(len(images)):
    image_c = images[img_idx]
# finding the topmost number in this image
    for i in range(10):
        if compare_hist(image_c, numbers[9 - i]):
            max_student_num.append(9 - i)
            break
# choosing 3 that I think in this case is the best for our case
q_img = quantization(images, n_colors=3)
# moving again over all the images
for img_idx in range(len(images)):
    # safely threshold the quantized image threshold=245 due the way that quantization func work
    _, binary_image = cv2.threshold(q_img[img_idx], 245, 255, cv2.THRESH_BINARY)

# getting the height of each bar in the image[img_idx], storing it in list(pixels_height)
    pixels_height = []
    for idx in range(10):
        pixels_height.append(get_bar_height(binary_image, idx))

    # saving the heights of each list in the list
    heights = []
    # getting the max bin heights by printing some image heights_pixels, we see that the max
    # bin height at that image is 153 (because our images is the same [height, width])
    max_bin_height = 154
    for idx in range(10):
        heights.append(round(max_student_num[img_idx] * pixels_height[idx] / max_bin_height))
    # printing the text version of each histogram.
    print(f'Histogram {names[img_idx]} gave {",".join(map(str, heights))}')

exit()
