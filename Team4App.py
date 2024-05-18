# -*- coding: utf-8 -*-
"""
Created on Thu May  16 20:11:05 2024
@author: bih13
"""
import pandas as pd
import streamlit as st
import pickle
from PIL import Image

# Load your trained model and vectorizer
filepath = 'Team4model_nb.pkl'
model = pd.read_pickle(filepath )
filepath2 = 'Team4vectorizer_pipeline.pkl'
vectorizer = pd.read_pickle(filepath2 )

def predict(email_text):
    processed_text = vectorizer.transform([email_text])
    prediction = model.predict(processed_text)
    return prediction[0]

# Display an image from a URL
image_url = "https://miro.medium.com/v2/resize:fit:1400/0*mbFBPcPUJD-53v3h.png"
st.image(image_url, caption="Spam Detection", use_column_width=True)

st.title("Team 4 Project")
st.markdown("### Email Spam Detection App")

email_input = st.text_area("Enter your email text and we will check it for you for free!:")

if st.button("Predict"):
    prediction = predict(email_input)
    st.write(f"Prediction: {prediction}")
