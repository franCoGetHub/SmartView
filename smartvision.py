import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input, decode_predictions

# Load pre-trained EfficientNetB7 model
model = EfficientNetB7(weights='imagenet')

def classify_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (600, 600))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    preds = model.predict(img)
    label = decode_predictions(preds, top=1)[0][0][1]
    return label

st.title("PetPals App")

st.write("Welcome to the PetPals Image Classification App")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Save the image to a temporary location
    temp_image_path = "/tmp/temp_image.jpg"
    image.save(temp_image_path)
    
    if st.button("Classify"):
        label = classify_image(temp_image_path)
        st.write(f"Predicted label: {label}")
