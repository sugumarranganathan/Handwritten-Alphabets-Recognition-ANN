import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("alphabet_model.h5")

labels=[chr(i) for i in range(65,91)]

st.title("Handwritten Alphabet Recognition using ANN")

uploaded_file=st.file_uploader(
    "Upload image",
    type=["png","jpg","jpeg"]
)

if uploaded_file is not None:

    img=Image.open(uploaded_file).convert('L')

    img=img.resize((28,28))

    img=np.array(img)

    img=img/255.0

    img=img.reshape(1,784)

    prediction=model.predict(img)

    pred=np.argmax(prediction)

    st.success(
        f"Predicted Alphabet: {labels[pred]}"
    )
