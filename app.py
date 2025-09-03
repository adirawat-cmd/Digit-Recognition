import streamlit as st
import numpy as np
import cv2
from tensorflow import keras


model = keras.models.load_model("firstmodel.h5")

st.title("Digit Classifier")
uploaded_file = st.file_uploader("Upload a digit image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    
    img = cv2.resize(img, (28,28))
    img = 255 - img
    img = img / 255.0
    img = img.reshape(1,28,28,1)

    prediction = model.predict(img)
    st.image(img.reshape(28,28), caption="Processed Image", width=150)
    st.write("Predicted Digit:", np.argmax(prediction))
