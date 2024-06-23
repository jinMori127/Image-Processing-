# Ayal_Kaabia, 322784760
# Sami_Serhan, 327876298

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import matplotlib.pyplot as plt



def get_laplacian_pyramid(image, levels, resize_ratio=0.5):
    """
    Constructs a Laplacian pyramid from the given image.

    Parameters:
        image (numpy.ndarray): The input image.
        levels (int): The number of levels in the pyramid.
        resize_ratio (float, optional): The ratio by which the image is resized at each level. Default is 0.5.

    Returns:
        list: A list containing the Laplacian pyramid levels.
    """
    pyramid = []  # Initialize an empty list to hold pyramid levels
    temp = np.float32(image.copy())  # Convert image to float32
    # Iterate over the specified number of levels
    for _ in range(levels - 1):
        # Apply Gaussian blur to the image to reduce noise
        blurred = cv2.GaussianBlur(temp, (5, 5), 0)
        # Calculate the destination size based on the original image size and the scaling factor
        destination_size = (int(temp.shape[1] * resize_ratio), int(temp.shape[0] * resize_ratio))
        # Check if the resize ratio is positive before performing resizing
        if resize_ratio > 0:
            # Downsample the blurred image using resizing with the destination size
            downsampled = cv2.resize(blurred, destination_size, interpolation=cv2.INTER_LINEAR)
        else:
            # If resize ratio is not positive, set downsampled to blurred (no resizing)
            downsampled = blurred
        # Upsample the downsampled image to the original size
        expanded = cv2.resize(downsampled, (temp.shape[1], temp.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Calculate the Laplacian by subtracting the expanded image from the original
        laplacian = temp - expanded
        # Append the Laplacian to the pyramid list
        pyramid.append(laplacian)
        # Update the temporary image for the next iteration
        temp = np.float32(downsampled)
    # Append the smallest level of Gaussian pyramid (original image) to the pyramid list
    pyramid.append(temp)
    return pyramid


def restore_from_pyramid(pyramidList, resize_ratio=2):
    """
    Reconstructs the original image from the Laplacian pyramid.

    Parameters:
        pyramidList (list): The Laplacian pyramid levels.
        resize_ratio (float, optional): The ratio by which the image is resized during reconstruction. Default is 2.

    Returns:
        numpy.ndarray: The reconstructed original image.
    """
    restored_image = pyramidList[-1]  # Initialize with the smallest level of the Laplacian pyramid
    # Iterate over the pyramid levels in reverse order, excluding the smallest level
    for i in range(len(pyramidList) - 2, -1, -1):
        # Upsample the previously restored image to match the current pyramid level size
        expanded = cv2.resize(restored_image, (pyramidList[i].shape[1], pyramidList[i].shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        # Add the upsampled image to the Laplacian pyramid level
        restored_image = cv2.add(expanded, pyramidList[i])
    return restored_image


def blend_pyramids(levels, pyr_orange, pyr_apple):
    """
    Blends two Laplacian pyramids based on a masking technique.

    Parameters:
        levels (int): The number of levels in the pyramid.
        pyr_orange (list): Laplacian pyramid of one image.
        pyr_apple (list): Laplacian pyramid of another image.

    Returns:
        list: A blended Laplacian pyramid.
    """
    blended_pyr = []
    for curr_level in range(levels):
        # Define mask in the size of the current pyramid level
        mask = np.zeros_like(pyr_orange[curr_level], dtype=np.float32)
		# Get level width for this current level
        level_width = pyr_orange[curr_level].shape[1]

        # Create the mask
        for col in range(int(0.5 * level_width - (curr_level))):
            mask[:, col] = 1.0

        for col in range(int(0.5 * level_width - (curr_level)), int(0.5 * level_width + (curr_level - 5))):
            mask[:, col] = 0.9 - 0.9 * (col - (0.5 * level_width - (curr_level))) / (2 * (curr_level))

        # Blend the pyramid level for curr_level
        blended_level_float = pyr_orange[curr_level] * mask + pyr_apple[curr_level] * (1 - mask)
		# Adding the blended level to our blended pyramid
        blended_pyr.append(blended_level_float)

    return blended_pyr



apple = cv2.imread('apple.jpg')
apple = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)

orange = cv2.imread('orange.jpg')
orange = cv2.cvtColor(orange, cv2.COLOR_BGR2GRAY)

# validate_operation(apple)
# validate_operation(orange)
levels=5
pyr_apple = get_laplacian_pyramid(apple,levels)
pyr_orange = get_laplacian_pyramid(orange,levels)

pyr_result = []
# Your code goes here
pyr_result=blend_pyramids(levels, pyr_orange, pyr_apple)

final = restore_from_pyramid(pyr_result)
plt.imshow(final, cmap='gray')
plt.show()

cv2.imwrite("result.jpg", final)

