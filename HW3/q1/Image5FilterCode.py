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


def main():
    # Open the image file
    image_path = "1.jpg"  # Update with your image file path
    try:
        image = cv2.imread(image_path)
    except FileNotFoundError:
        print("Image file not found.")
        return

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    filtered_image=grayscale_image-cv2.GaussianBlur(grayscale_image,(15,15),4)+127
    # Display the filtered image
    cv2.imshow("Gausian Filtered Image 5", filtered_image)
    cv2.waitKey(0)

    # Save the filtered image
    cv2.imwrite("fixed_image_5.jpg", filtered_image)

    # Open the image file
    image_path = "image_5.jpg"  # Update with your image file path
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
