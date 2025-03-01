from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Create folders
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load Models
classification_model = tf.keras.models.load_model("models/densenet_covid_model.keras")
infection_segmentation_model = tf.keras.models.load_model("models/unet_infection_segmentation_model1.keras")
lung_segmentation_model = tf.keras.models.load_model("models/unet_lung_segmentation_model.keras")

# Image size
IMG_WIDTH = 256
IMG_HEIGHT = 256

# Load and resize the image
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    return image

# Generate a class prediction with segmentation and save
def predict_and_save(image_path):
    image = load_image(image_path)
    expanded_image = np.expand_dims(image, axis=0)

    # Predict classification
    prediction = classification_model.predict(expanded_image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Lung Segmentation
    lung_mask = lung_segmentation_model.predict(expanded_image)[0]
    lung_mask = np.repeat(lung_mask, 3, axis=-1)  # Shape: (256, 256, 3)
    segmented_image = image.copy()
    segmented_image[lung_mask == 0] = 0


    plt.figure(figsize=(10, 5))

    if predicted_class == 0:  # COVID case
        infection_mask = infection_segmentation_model.predict(expanded_image)[0]

        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.xlabel("Original Image", fontsize=16)
        plt.xticks([]), plt.yticks([])

        plt.subplot(1, 3, 2)
        plt.imshow(segmented_image)
        plt.xlabel("Lung Segmentation", fontsize=16)
        plt.xticks([]), plt.yticks([])

        plt.subplot(1, 3, 3)
        plt.imshow(segmented_image)
        plt.imshow(infection_mask, 'Reds', alpha=0.4)
        plt.xlabel("Infection Segmentation", fontsize=16)
        plt.suptitle(f"COVID Detected (Confidence: {np.max(prediction):.2f})", fontsize=20)
        plt.xticks([]), plt.yticks([])

    else:
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.xlabel("Original Image", fontsize=16)
        plt.xticks([]), plt.yticks([])

        plt.subplot(1, 2, 2)
        plt.imshow(segmented_image)
        plt.xlabel("Lung Segmentation", fontsize=16)
        if predicted_class == 1: # Non-COVID case
            plt.suptitle(f"Non-COVID Condition (Confidence: {np.max(prediction):.2f})", fontsize=20)
        else: # Normal case
            plt.suptitle(f"Normal Condition (Confidence: {np.max(prediction):.2f})", fontsize=20)
        plt.subplots_adjust(top=0.85)
        plt.xticks([]), plt.yticks([])

    #Save prediciton as image
    result_path = os.path.join(RESULT_FOLDER, "Detection_result.png")
    plt.savefig(result_path, bbox_inches='tight')
    plt.close()

    return result_path

#Route the prediction
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['image']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    result_path = predict_and_save(file_path)

    return jsonify({'result_url': f'/results/Detection_result.png'})

# Get the prediciton image
@app.route('/results/<filename>')
def get_result_image(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
