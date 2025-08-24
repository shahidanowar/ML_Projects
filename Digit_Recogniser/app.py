import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('digit_model.keras')

st.title('Handwritten Digit Recognizer')
st.write('Draw a digit below and click predict.')

# Create a canvas for drawing
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button('Predict'):
    if canvas_result.image_data is not None:
        # Get the image data from the canvas
        img = canvas_result.image_data.astype('uint8')

        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize to 28x28 for the model
        img_resized = cv2.resize(img, (28, 28))

        # Normalize the image to be in the range [0, 1]
        img_normalized = tf.keras.utils.normalize(img_resized, axis=1)

        # Reshape for the model
        img_reshaped = np.reshape(img_normalized, (1, 28, 28))

        # Predict the digit
        prediction = model.predict(img_reshaped)
        predicted_digit = np.argmax(prediction)

        st.write(f'Predicted Digit: {predicted_digit}')
    else:
        st.write('Please draw a digit first.')