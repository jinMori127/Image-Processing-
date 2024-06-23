import numpy as np
from PIL import Image

def average_filter_row_wise(image):
    """Apply average filter row-wise to the input grayscale image."""
    # Convert image to numpy array
    img_array = np.array(image)

    # Get image dimensions
    height, width = img_array.shape

    # Create a new array to store the filtered image
    filtered_img_array = np.zeros((height, width), dtype=np.uint8)

    # Apply average filter row-wise
    for y in range(height):
        # Compute the average pixel value for each row
        average_value = np.mean(img_array[y, :])
        # Assign the average value to all pixels in the row
        filtered_img_array[y, :] = average_value

    # Convert the filtered numpy array back to PIL Image
    filtered_image = Image.fromarray(filtered_img_array)

    return filtered_image

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

    # Apply average filter row-wise
    filtered_image = average_filter_row_wise(grayscale_image)

    # Display the filtered image
    filtered_image.show(title="Filtered Image (Row-wise Average)")

    filtered_image.save("fixed_image_1.jpg")
    # Open the image file
    image_path = "image_1.jpg"  # Update with your image file path
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print("Image file not found.")
        return

    # Calculate Mean Squared Error (MSE)
    mse = calculate_mse(np.array(image), np.array(filtered_image))
    print("Mean Squared Error (MSE) between original and filtered image:", mse)

if __name__ == "__main__":
    main()
