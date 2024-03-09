import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import pandas as pd

def detect_glaucoma(uploaded_file):
    # Convert the uploaded file to a PIL Image object
    pil_image = Image.open(uploaded_file)
    
    # Convert the PIL Image to a NumPy array
    image_np = np.array(pil_image)
    
    # Check if the image is successfully loaded
    if image_np is None:
        print("Error: Unable to load the image.")
        return None

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image_np, (65, 65), 0)

    # Find the pixel with the highest intensity value
    max_intensity_pixel = np.unravel_index(np.argmax(blurred_image), blurred_image.shape)

    # Define the radius for the circle
    radius = 200 // 2

    # Get the x and y coordinates for cropping the image
    x = max_intensity_pixel[1] - radius
    y = max_intensity_pixel[0] - radius

    # Create a mask for the circle
    mask = np.zeros_like(image_np)
    cv2.circle(mask, (x + radius, y + radius), radius, (255, 255, 255), -1)

    # Apply the mask to the original image
    roi_image = cv2.bitwise_and(image_np, mask)

    # Split the green channel
    green_channel = roi_image[:, :, 1]
    
    # Apply histogram equalization
    clahe_op = cv2.createCLAHE(clipLimit=2)
    roi_image = clahe_op.apply(green_channel)

    # Save the preprocessed image
    # cv2.imwrite("image.jpg", roi_image)

    # Load the pre-trained model
    model = load_model("/workspaces/GLAUCOMA_DETECTIONS/src/detect_glaucoma/GLAUCOMA_DETECTION.h5")
    model.compile(loss='binary_crossentropy',
              optimizer=Adam(1e-4),
              metrics=['binary_accuracy'])

    # Resize the image to (256, 256)
    input_image = cv2.resize(roi_image, (256, 256))

    # Create an instance of ImageDataGenerator for preprocessing
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)  # You can add other preprocessing options here

    # Create a DataFrame with the path to the image you want to predict on
    image_path_df = pd.DataFrame({'image_path': ['image.jpg']})

    # Use flow_from_dataframe to generate batches of images for prediction
    # You need to specify the target_size and batch_size
    image_generator = datagen.flow_from_dataframe(
        dataframe=image_path_df,
        x_col='image_path',
        y_col=None,  # For prediction, y_col can be set to None
        target_size=(256, 256),  # Specify the target size of the images
        batch_size=1,  # Set batch_size to 1 for predicting on a single image
        class_mode=None,  # For prediction, class_mode can be set to None
        shuffle=False , # No need to shuffle since there's only one image
        verbose = 0
    )
#
    prediction = model.predict(image_generator)
     # Define thresholds for classifying predictions
    # Define thresholds for classifying predictions
    threshold_glauc = 0.7
    threshold_mild_glauc = 0.4

    # Map predicted probabilities to categories
    if prediction >= threshold_glauc:
        category = "GLAUCOMA"
    elif prediction >= threshold_mild_glauc:
        category = "MILD GLAUCOMA"
    else:
        category = "NO GLAUCOMA"
    
    # Return a tuple containing the category and the prediction score
    return category, prediction[0][0]
