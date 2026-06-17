import streamlit as st
import numpy as np
import joblib
import cv2
import tempfile

from PIL import Image
from skimage.measure import shannon_entropy

# Load model and scaler
model = joblib.load("random_forest.pkl")
scaler = joblib.load("scaler.pkl")

# Feature Extraction
def extract_features(image_path):

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    width = img.width
    height = img.height

    aspect_ratio = width / height
    pixel_count = width * height

    mean_r = np.mean(img_np[:,:,0]) / 255
    mean_g = np.mean(img_np[:,:,1]) / 255
    mean_b = np.mean(img_np[:,:,2]) / 255

    std_r = np.std(img_np[:,:,0]) / 255
    std_g = np.std(img_np[:,:,1]) / 255
    std_b = np.std(img_np[:,:,2]) / 255

    entropy = shannon_entropy(img_np)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    edge_density = np.sum(edges > 0) / edges.size

    return [[
        width,
        height,
        aspect_ratio,
        pixel_count,
        mean_r,
        mean_g,
        mean_b,
        std_r,
        std_g,
        std_b,
        entropy,
        edge_density
    ]]

# Prediction
def predict_image(image_path):

    features = extract_features(image_path)

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    confidence = max(
        model.predict_proba(features_scaled)[0]
    ) * 100

    return prediction, confidence

# Streamlit UI
st.set_page_config(
    page_title="AI Art Detector",
    layout="centered"
)

st.title("🎨 AI Art vs Human Art Detector")

st.write(
    "Upload an image and the model will predict whether it is AI-generated or human-created."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as tmp:

        tmp.write(uploaded_file.getvalue())

        temp_path = tmp.name

    prediction, confidence = predict_image(temp_path)

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success(
            f"AI Generated Art\nConfidence: {confidence:.2f}%"
        )

    else:

        st.success(
            f"Human Created Art\nConfidence: {confidence:.2f}%"
        )