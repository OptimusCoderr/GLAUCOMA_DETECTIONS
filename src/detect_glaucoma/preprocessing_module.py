import cv2
import numpy as np

def custom_preprocessing(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Check if the image is successfully loaded
    if image is None:
        print("Error: Unable to load the image.")
        return None

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (65, 65), 0)

    # Find the pixel with the highest intensity value
    max_intensity_pixel = np.unravel_index(np.argmax(blurred_image), blurred_image.shape)

    # Define the radius for the circle
    radius = 200 // 2

    # Get the x and y coordinates for cropping the image
    x = max_intensity_pixel[1] - radius
    y = max_intensity_pixel[0] - radius

    # Create a mask for the circle
    mask = np.zeros_like(image)
    cv2.circle(mask, (x + radius, y + radius), radius, (255, 255, 255), -1)

    # Apply the mask to the original image
    roi_image = cv2.bitwise_and(image, mask)

    # Split the green channel
    green_channel = roi_image[:, :, 1]
    
    # Apply histogram equalization
    clahe_op = cv2.createCLAHE(clipLimit=2)
    roi_image = clahe_op.apply(green_channel)

    return roi_image
