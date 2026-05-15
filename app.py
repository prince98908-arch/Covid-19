import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load Trained Model
model = tf.keras.models.load_model('vgg16_covid_model.h5')

# Class Names
classes = ['Covid', 'Normal', 'Viral Pneumonia']

# App Title
st.title('COVID-19 Detection from Chest X-Ray')

st.write('Upload a Chest X-ray image to predict disease type.')

# File Upload
uploaded_file = st.file_uploader(
    'Choose a Chest X-ray Image',
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:

    # Open Image
    image = Image.open(uploaded_file)

    # Display Image
    st.image(
        image,
        caption='Uploaded Image',
        use_column_width=True
    )

    # Convert Image to Array
    img = np.array(image)

    # Resize Image
    img = cv2.resize(img, (128, 128))

    # Normalize
    img = img / 255.0

    # Expand Dimensions
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)

    predicted_class = classes[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    # Display Result
    st.success(f'Prediction: {predicted_class}')

    st.info(f'Confidence: {confidence:.2f}%')
