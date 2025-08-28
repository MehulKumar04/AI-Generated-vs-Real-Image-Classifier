import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("att_model.keras")

# Define class labels
CLASS_NAMES = ["Real Art", "AI Generated"]

# Preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))   # Resize to match model input
    image = np.array(image) / 255.0    # Normalize to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("🎨 AI Generated vs Real Art Classifier")
st.write("Upload an image and the model will classify it as **AI Generated** or **Real Art**.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess & predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]

    # Classification
    if prediction > 0.5:
        st.success(f"🖼️ Prediction: **AI Generated** (Confidence: {prediction:.2f})")
    else:
        st.success(f"🖼️ Prediction: **Real Art** (Confidence: {1 - prediction:.2f})")
