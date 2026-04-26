import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model

model = load_model("digits_model.keras", compile=False)

st.title("✍️ Handwritten Digit Recognition")

uploaded_file = st.file_uploader("Upload digit image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
  # Show image
  img = Image.open(uploaded_file)
  st.image(img, caption="Uploaded Image", width=250)
  # Preprocessing (same as testing)
  img = img.convert('L')
  img = img.resize((28, 28))
  img_array = np.array(img)
  img_array = img_array / 255.0
  img_array = img_array.reshape(1, 28, 28, 1)
  # Show processed image
  st.image(img_array.reshape(28,28), caption="Processed Image", width=150)
  # Prediction
  prediction = model.predict(img_array)
  predicted_class = np.argmax(prediction)
  confidence = np.max(prediction)
  # Output
  st.success(f"Predicted Digit: {predicted_class}")
  st.write(f"Confidence: {confidence*100:.2f}%")
