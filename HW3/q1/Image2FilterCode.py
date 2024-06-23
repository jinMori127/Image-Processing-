from PIL import Image
import numpy as np
from scipy.ndimage import convolve


def average_filter(image, kernel_size=3):
    """Apply an Average/Mean/Smoothing filter to the input image."""
    # Define the averaging kernel
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

    # Apply convolution with the averaging kernel
    smoothed_image = convolve(image, kernel)

    return smoothed_image

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

    # Apply Average/Smoothing filter
    kernel_size = 11  # Adjust kernel size as needed
    smoothed_image = average_filter(np.array(grayscale_image), kernel_size)

    # Convert the smoothed image back to PIL Image
    smoothed_image_pil = Image.fromarray(smoothed_image.astype(np.uint8))



    # Display the smoothed image
    smoothed_image_pil.show(title=f"Smoothed Image (Kernel Size={kernel_size})")

    # Save the smoothed image
    smoothed_image_pil.save("fixed_image_2.jpg")  # Save the smoothed image

    # Open the image file
    image_path = "image_2.jpg"  # Update with your image file path
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print("Image file not found.")
        return

    # Calculate Mean Squared Error (MSE)
    mse = calculate_mse(np.array(image), np.array(smoothed_image_pil))
    print("Mean Squared Error (MSE) between original and filtered image:", mse)


if __name__ == "__main__":
    main()
