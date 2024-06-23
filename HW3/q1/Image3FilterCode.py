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

    # Apply median filter
    radius = 9  # Radius of the median filter
    median_filtered_image = grayscale_image.filter(ImageFilter.MedianFilter(radius))

    # Display the median filtered image
    median_filtered_image.show(title=f"Median Filtered Image (Radius={radius})")

    # Save the median filtered image
    median_filtered_image.save("fixed_image_3.jpg")  # Save the filtered image

    # Open the image file
    image_path = "image_3.jpg"  # Update with your image file path
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print("Image file not found.")
        return
    # Calculate Mean Squared Error (MSE)
    mse = calculate_mse(np.array(image), np.array(median_filtered_image))
    print("Mean Squared Error (MSE) between original and filtered image:", mse)

if __name__ == "__main__":
    main()
