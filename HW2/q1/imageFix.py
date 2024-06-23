# Ayal_Kaabia, 322784760
# Sami_Serhan, 327876298

# Please replace the above comments with your names and ID numbers in the same format.
import cv2
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import numpy as np


def apply_fix(image, id):
    # Adjust brightness and contrast
    alpha = 2.5  # Contrast control (1.0-3.0)
    beta = 80  # Brightness control (0-100)
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image


for i in range(1, 4):
    path = f'{i}.jpg'
    if i==1:
        path= f'{i}.png'
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Unable to read the image {path}")
        continue

    fixed_image = apply_fix(image, i)
    plt.imsave(f'{i}_fixedBrightnessNew.jpg', fixed_image, cmap='gray')


