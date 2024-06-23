import cv2
import numpy as np
from PIL import Image


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


# Load the original image
original_image = cv2.imread('1.jpg')

# Convert the original image to grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Split the grayscale image into upper and lower halves
height, width = gray_image.shape[:2]
upper_half = gray_image[0:height//2, :]
lower_half = gray_image[height//2:, :]

# Create a new blank image of the same size as the original image
new_image = 255 * np.ones_like(original_image)

# Paste the upper half of the grayscale image onto the lower half of the new image
new_image[height//2:, :] = cv2.cvtColor(upper_half, cv2.COLOR_GRAY2BGR)

# Paste the lower half of the grayscale image onto the upper half of the new image
new_image[0:height//2, :] = cv2.cvtColor(lower_half, cv2.COLOR_GRAY2BGR)

# Display the new image
cv2.imshow('New Image', new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the new image
cv2.imwrite('fixed_image_7.jpg', new_image)
#CALCULATING MSE
new_image = cv2.imread('fixed_image_7.jpg')
# Convert images to grayscale
new_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
# Open the image file
image_path = "image_7.jpg"  # Update with your image file path
try:
    image = Image.open(image_path)
except FileNotFoundError:
        print("Image file not found.")


# Calculate Mean Squared Error (MSE)
mse = calculate_mse(np.array(image), np.array(new_gray))
print("Mean Squared Error (MSE) between original and filtered image:", mse)
