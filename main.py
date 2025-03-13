import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Load the saved CNN model
model = keras.models.load_model('cifar10_cnn_model.h5')

# Class labels for CIFAR-10
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Streamlit app
st.title("CIFAR-10 Image Classification 🔼️")
st.write("Upload an image, and the CNN model will predict its class.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        # Open and process the image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)

        # Preprocess the image
        image_resized = image.resize((32, 32))
        image_array = np.array(image_resized) / 255.0
        image_batch = np.expand_dims(image_array, axis=0)

        # Prediction with loading spinner
        if st.button("Predict"):
            with st.spinner('Predicting...'):
                predictions = model.predict(image_batch)
                predicted_class = np.argmax(predictions)
                confidence = np.max(predictions) * 100

                st.subheader("Prediction Results")
                st.write(f"**Predicted Class:** {labels[predicted_class]}")
                st.write(f"**Confidence:** {confidence:.2f}%")

                # Check for low confidence
                if confidence < 50:
                    st.warning("The model is not very confident about this prediction. Try another image.")

                # Display the top 3 predictions
                st.write("**Top 3 Predictions:**")
                top_3_indices = np.argsort(predictions[0])[::-1][:3]
                for i in top_3_indices:
                    st.write(f"- {labels[i]}: {predictions[0][i] * 100:.2f}%")
    
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
