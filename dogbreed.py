# -*- coding: utf-8 -*-

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import json

#title_img = Image.open('allcartoons_bg.jpg')

#st.image(title_img, width=720)

IMAGE_SIZE = [64, 64]

st.header('Dog breed Classifier')
st.subheader("Image")
uploaded_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
model = load_model('dogbreed_vgg19.h5')
# st.write(image)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Input', width=250)
    img_array = np.array(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)    
    img = cv2.resize(img,tuple(IMAGE_SIZE))
    img = np.reshape(img,[1,64,64,3])
    classes = model.predict(img)

    index = np.argmax(classes)
    with open('data.json', 'r') as fp:
        data = json.load(fp)

    st.success(data[str(index)])


#cv2.imwrite('out.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))