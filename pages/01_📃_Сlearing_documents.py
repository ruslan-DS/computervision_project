import urllib.request
import streamlit as st
import matplotlib.pyplot as plt 
import numpy as np
import torch
from PIL import Image
from model.model import  ConvAutoencoder
from weights.preprocessing import preprocess

device = 'cpu'

@st.cache_resource
def load_model():
    model = ConvAutoencoder()
    model.load_state_dict(torch.load('weights/best_weights_v5.pt', map_location = 'cpu'))
    model.eval()
    return model.to(device)

model = load_model()

st.title('Очищение документов от шума')

def predict(img):
    img = preprocess(img)
    img.to(device)
    outputs = model(img.unsqueeze(0))
    pred = outputs.detach().cpu().squeeze(0).numpy()
    return pred

col1, col2 = st.columns(2)

selected = st.radio('Метод загрузки', ['Файлы', 'URL-адрес'])

# Process uploaded files
if selected == 'Файлы':
    uploaded_files = st.file_uploader('Загрузите изображения документов', accept_multiple_files=True)    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file)
            prediction = predict(img)
            
            with col1:
                st.markdown('<p style="font-family: Arial; font-weight: bold;">Shabby document</p>', unsafe_allow_html=True)
                st.image(img)
            with col2:
                st.markdown('<p style="font-family: Helvetica; font-weight: bold;">Сleared document</p>', unsafe_allow_html=True)
                st.image(prediction[0])

# Process image URLs
else:
    input_urls = st.text_area('Введите URL-ы изображений документов (разделяйте их новой строкой)')
    if input_urls:
        urls = input_urls.split('\n')
        for url in urls:
            try:
                response = urllib.request.urlopen(url)
                img = Image.open(response)
                prediction = predict(img)
                
                with col1:
                    st.markdown('<p style="font-family: Arial; font-weight: bold;">Shabby document</p>', unsafe_allow_html=True)
                    st.image(img)
                with col2:
                    st.markdown('<p style="font-family: Helvetica; font-weight: bold;">Сleared document</p>', unsafe_allow_html=True)
                    st.image(prediction[0])
            except:
                st.error(f"Failed to load image from URL: {url}")
