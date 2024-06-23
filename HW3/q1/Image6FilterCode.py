import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import convolve

def horizontal_laplacian_filter(image):
    """Apply horizontal Laplacian filter to the input grayscale image."""
    # Convert image to numpy array
    img = np.array(image)

    kernel = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])
    return cv2.filter2D(img,-1,0.33*kernel)


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

def main():
    # Open the image file
    image_path = "1.jpg"  # Update with your image file path
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print("Image file not found.")
        return

    # Convert the image to grayscale
    grayscale_image = image.convert("L")

    # Apply horizontal Laplacian filter
    horizontal_laplacian_filtered_image = horizontal_laplacian_filter(grayscale_image)
    # Convert the NumPy array back to a PIL Image object
    horizontal_laplacian_filtered_image_pil = Image.fromarray(horizontal_laplacian_filtered_image)

    # Display the filtered image
    horizontal_laplacian_filtered_image_pil.show(title="Horizontal Laplacian Filtered Image")

    # Save the filtered image
    horizontal_laplacian_filtered_image_pil.save("fixed_image_6.jpg")
    # Open the image file
    image_path = "image_6.jpg"  # Update with your image file path
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print("Image file not found.")
        return
    # Calculate Mean Squared Error (MSE)
    mse = calculate_mse(np.array(image), np.array(horizontal_laplacian_filtered_image))
    print("Mean Squared Error (MSE) between original and filtered image:", mse)

if __name__ == "__main__":
    main()
