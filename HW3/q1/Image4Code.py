import cv2
import numpy as np
from PIL import Image, ImageFilter
def calculate_mse(original_image, filtered_image):
    """Calculate Mean Squared Error (MSE) between original and filtered images."""
    # Convert images to numpy arrays
    original_array = np.array(original_image)
    filtered_array = np.array(filtered_image)

    # Calculate squared differences between corresponding pixels
    squared_diff = (original_array - filtered_array) ** 2

    # Compute the mean squared error
    mse = np.mean(squared_diff)

    return mse


def vertical_average_filter(image, kernel_size):
    """Apply vertical average filter to the input grayscale image."""
    # Convert image to grayscale if not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    # Apply vertical average filter
    blurred_image = cv2.blur(gray_image, (1, kernel_size))

    return blurred_image

def main():
    # Open the image file
    image_path = "1.jpg"  # Update with your image file path
    try:
        image = cv2.imread(image_path)
    except FileNotFoundError:
        print("Image file not found.")
        return

    # Apply vertical average filter
    kernel_size = 14  # Adjust the kernel size to control the blur intensity
    blurred_image = vertical_average_filter(image, kernel_size)



    # Display the blurred image
    cv2.imshow("Blurred Image (Vertical Average Filter)", blurred_image)
    cv2.waitKey(0)

    # Save the blurred image
    cv2.imwrite("fixed_image_4.jpg", blurred_image)

    # Open the image file
    image_path = "image_4.jpg"  # Update with your image file path
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print("Image file not found.")
        return

    # Calculate Mean Squared Error (MSE)
    mse = calculate_mse(np.array(image), np.array(blurred_image))
    print("Mean Squared Error (MSE) between original and filtered image:", mse)

if __name__ == "__main__":
    main()
