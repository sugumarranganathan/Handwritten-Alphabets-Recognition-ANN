import streamlit as st
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# 1. Safe path loading: Ensures the file is found in the deployment directory
model_path = os.path.join(os.path.dirname(__file__), "alphabet_model.h5")

# 2. Cache the model: Prevents reloading the heavy model on every user interaction
@st.cache_resource
def load_model_cached():
    return load_model(model_path)

# Load the model once
model = load_model_cached()

labels = [chr(i) for i in range(65, 91)]  # Generates A-Z

st.title("Handwritten Alphabet Recognition using ANN")

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert to Grayscale
    img = Image.open(uploaded_file).convert('L')
    
    # Resize to 28x28
    img = img.resize((28, 28))
    
    # Convert to Numpy array and Normalize (0-1)
    img_array = np.array(img)
    img_array = img_array / 255.0
    
    # Flatten for ANN input (784 pixels)
    img_array = img_array.reshape(1, 784)
    
    # Predict
    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction)
    
    # Display Result
    st.success(f"Predicted Alphabet: **{labels[pred_index]}**")
