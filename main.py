import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array


model = load_model('mymodel.h5')

st.title("DMD Scooter Classification App!")

st.image("https://www.intelligenttransport.com/wp-content/uploads/Lime-3.jpg", use_column_width=True)

file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])


def predict_label(img):
    img = img.resize((150, 300))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return np.argmax(prediction)


def get_class_name(prediction):
    if prediction == 0:
        return "Scooter recognized"
    elif prediction == 1:
        return "Not a Scooter"
    elif prediction == 2:
        return "Scooter Not Parked Properly"
    else:
        return "Scooter Blocks Footpath"


if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, use_column_width=True)
    prediction = predict_label(image)
    prediction_name = get_class_name(prediction)
    st.write(f"Prediction: {prediction_name} \n\n Label {prediction}")
