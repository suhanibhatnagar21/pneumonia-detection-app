
import streamlit as st
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)


# ---------------------------------------------------------
# Load trained model from Hugging Face
# ---------------------------------------------------------

@st.cache_resource
def load_pneumonia_model():

    model_path = hf_hub_download(
        repo_id="Suhani2128/pneumonia-detection-model",
        filename="best_pneumonia_model.keras"
    )

    return load_model(model_path)


model = load_pneumonia_model()


# ---------------------------------------------------------
# Application title
# ---------------------------------------------------------

st.title("🫁 Pneumonia Detection")

st.write(
    "Upload a chest X-ray image to predict whether the image "
    "is classified as NORMAL or PNEUMONIA."
)

st.warning(
    "This application is an academic machine-learning prototype "
    "and is not intended for clinical diagnosis."
)


# ---------------------------------------------------------
# Image upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded X-ray")

    st.image(
        image,
        caption="Uploaded chest X-ray",
        use_container_width=True
    )

    # Same preprocessing used during notebook inference
    image_resized = image.resize((224, 224))

    image_array = np.array(image_resized) / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Make prediction
    probability = float(
        model.predict(
            image_array,
            verbose=0
        )[0][0]
    )

    if probability >= 0.5:

        prediction = "PNEUMONIA"
        confidence = probability

    else:

        prediction = "NORMAL"
        confidence = 1 - probability


    # -----------------------------------------------------
    # Display prediction
    # -----------------------------------------------------

    st.subheader("Prediction")

    if prediction == "PNEUMONIA":

        st.error(
            f"Prediction: **{prediction}**"
        )

    else:

        st.success(
            f"Prediction: **{prediction}**"
        )


    st.metric(
        "Prediction Confidence",
        f"{confidence * 100:.2f}%"
    )

    st.write(
        f"Probability of Pneumonia: "
        f"**{probability * 100:.2f}%**"
    )

    st.write(
        f"Probability of Normal: "
        f"**{(1 - probability) * 100:.2f}%**"
    )

