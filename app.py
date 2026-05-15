import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import zipfile
import os

# Extract ZIP File
if not os.path.exists("vgg16_covid_model.h5"):

    with zipfile.ZipFile("vgg16_covid_model.zip", 'r') as zip_ref:
        zip_ref.extractall()

# Load Model
model = tf.keras.models.load_model("vgg16_covid_model.h5")

# Class Labels
classes = ['Covid', 'Normal', 'Viral Pneumonia']

# Title
st.title("COVID-19 Detection from Chest X-Ray")

# Upload Image
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption='Uploaded Image',
        use_column_width=True
    )

    # Preprocessing
    img = np.array(image)

    img = cv2.resize(img, (128,128))

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)

    predicted_class = classes[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    # Result
    st.success(f"Prediction: {predicted_class}")

    st.info(f"Confidence: {confidence:.2f}%")
