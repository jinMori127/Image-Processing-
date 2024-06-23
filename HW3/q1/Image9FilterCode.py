import cv2
import numpy as np
from PIL import Image

def laplacian_filter(image):
    """Apply horizontal Laplacian filter to the input grayscale image."""
    # Convert image to numpy array
    img = np.array(image)

    kernel = np.array([
        [0, -0.65, 0],
        [-0.65, 3.6, -0.65],
        [0, -0.65, 0]
    ])
    return cv2.filter2D(img,-1,kernel)
# Load the original image
original_image = cv2.imread('1.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Apply the Laplacian filter
laplacian = laplacian_filter(gray_image)



cv2.imwrite('fixed_image_9.jpg', laplacian)

# Open the image file
image_path = "image_9.jpg"  # Update with your image file path
try:
    image = Image.open(image_path)
except FileNotFoundError:
        print("Image file not found.")
# Compute Mean Squared Error (MSE)
mse = np.mean((image - laplacian) ** 2)
print(f"Mean Squared Error (MSE) between original and sharpened images: {mse}")

# Display the sharpened image

cv2.imshow('Sharpened Image', laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()
