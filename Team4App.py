# -*- coding: utf-8 -*-
"""
Created on Thu May  16 20:11:05 2024
@author: bih13
"""

import pandas as pd
import streamlit as st
import requests
import joblib

@st.cache(allow_output_mutation=True)
def download_file(url, filename):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error downloading file: {url}")
    with open(filename, "wb") as file:
        file.write(response.content)
    return filename

# URLs of the files in the GitHub repository
pipeline_url = "https://raw.githubusercontent.com/BaderIAlharbi/Team4/main/Team_4_vectorizer.pkl"
model1_url = "https://raw.githubusercontent.com/BaderIAlharbi/Team4/main/Team_4_knn_spam_detection_model_nb.pkl"
model2_url = "https://raw.githubusercontent.com/BaderIAlharbi/Team4/main/Team_4_nb_spam_detection_model_nb.pkl"
image_url = "https://miro.medium.com/v2/resize:fit:1400/0*mbFBPcPUJD-53v3h.png" 

try:
    # Download and load the pipeline and model
    pipeline_path = download_file(pipeline_url, "Team_4_vectorizer.pkl")
    model1_path = download_file(model1_url, "Team_4_knn_spam_detection_model_nb.pkl")
    model2_path = download_file(model2_url, "Team_4_nb_spam_detection_model_nb.pkl")

    vectorizer = joblib.load(pipeline_path)
    model1 = joblib.load(model1_path)
    model2 = joblib.load(model2_path)
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

# Streamlit app layout with Background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://raw.githubusercontent.com/BaderIAlharbi/Team4/main/AdobeStock_376092029_Preview.jpeg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .custom-text-label {
        color: green;  /* Text color */
        background-color: yellow;  /* Highlight color */
        font-size: 18px;  /* Adjust the font size if needed */
        padding: 5px;  /* Add some padding for better readability */
        display: inline-block;  /* Ensure the highlight doesn't stretch */
    }
    .highlight-text {
        background-color: black;  /* Highlight color */
        color: white;  /* Text color to contrast with black background */
        padding: 5px;  /* Add some padding for better readability */
        display: inline-block;  /* Ensure the highlight doesn't stretch */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app layout
#st.title("Team 4 Project")
# Title with black highlight
st.markdown('<h1 class="highlight-text">Team 4 Project</h1>', unsafe_allow_html=True)
#st.markdown("### Email Spam Detection App")
# Header with black highlight
st.markdown('<h2 class="highlight-text">Email Spam Detection App</h2>', unsafe_allow_html=True)

st.markdown('<label class="custom-text-label">Enter your email text and we will check it for you for free!:</label>', unsafe_allow_html=True)
input_text = st.text_area("", "")

#input_text = st.text_area("Enter your email text and we will check it for you for free!:", "")

if st.button("Predict"):
    if input_text:
        try:
            processed_text = vectorizer.transform([input_text])
            prediction1 = model1.predict(processed_text)
            prediction2 = model2.predict(processed_text)
            #result = "Spam" if prediction[0] == 1 else "Not Spam"
            #st.write(f"Prediction: {prediction}")
            #prediction = "Spam"  # Example prediction value
            st.markdown(f'<p class="custom-text-label">Prediction via KNN: {prediction1}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="custom-text-label">Prediction via NB: {prediction2}</p>', unsafe_allow_html=True)
        except Exception as e:
            #st.error(f"Error during prediction: {e}")
            st.markdown(f'<p class="custom-text-label">Error during prediction: {e}</p>', unsafe_allow_html=True)
    else:
        #st.write("Please enter a message to predict.")
        st.markdown(f'<p class="custom-text-label">Please enter a message to predict, come on!</p>', unsafe_allow_html=True)

# Display the image
st.image(image_url, use_column_width=True)

# Disclaimer
st.markdown(f'<p class="custom-text-label">Disclaimer: This is an academic project based on a limited dataset of only 5,572 emails. Please do not rely on this tool for any critical decisions. Cheers!</p>', unsafe_allow_html=True)
