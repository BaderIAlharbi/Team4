# -*- coding: utf-8 -*-
"""
Created on Thu May  16 20:11:05 2024
@author: bih13
"""
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
import joblib
import requests
import os

@st.cache(allow_output_mutation=True)
def download_file(url, filename):
    response = requests.get(url)
    with open(filename, "wb") as file:
        file.write(response.content)
    return filename

# URLs of the files in the GitHub repository
pipeline_url = "https://raw.githubusercontent.com/BaderIAlharbi/Team4/Team4vectorizer_pipeline.pkl"
model_url = "https://raw.githubusercontent.com/BaderIAlharbi/Team4/Team4model_nb.pkl"

# Download and load the pipeline and model
pipeline_path = download_file(pipeline_url, "Team4vectorizer_pipeline.pkl")
model_path = download_file(model_url, "Team4model_nb.pkl")

'''
# Download the pipeline file
pipeline_response = requests.get(pipeline_url)
pipeline_path = "Team4vectorizer_pipeline.pkl"
with open(pipeline_path, "wb") as file:
    file.write(pipeline_response.content)

# Download the model file
model_response = requests.get(model_url)
model_path = "Team4model_nb.pkl"
with open(model_path, "wb") as file:
    file.write(model_response.content)
'''

# Load the pipeline and model using joblib
vectorizer = joblib.load(pipeline_path)
model = joblib.load(model_path)

# Load your trained model and vectorizer
#filepath = 'Team4model_nb.pkl'
#model = pd.read_pickle(filepath )
#filepath2 = 'Team4vectorizer_pipeline.pkl'
#vectorizer = pd.read_pickle(filepath2 )

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
