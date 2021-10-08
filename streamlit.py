import tensorflow as tf
model = tf.keras.models.load_model('model_car_damage.h5')
import streamlit as st
st.write("""
         # upload car image
         """
         )
st.write("This is a simple image classification web app to predict type of car damage")
file = st.file_uploader("Please upload an image file", type=["jpg", "png","jpeg"])

import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    
        size = (150,150)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(224, 224),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("bumper dent!")
    elif np.argmax(prediction) == 1:
        st.write("bumper scratch!")
    elif np.argmax(prediction) == 2:
        st.write("door dent!")
    elif np.argmax(prediction) == 3:
        st.write("door_scratch!")
    elif np.argmax(prediction) == 4:
        st.write("glass_shatter!")
    elif np.argmax(prediction) == 5:
        st.write("head_lamp!")
    elif np.argmax(prediction) == 6:
        st.write("multiple_damage!")
    
    else:
        st.write("tail_lamp!")
    
    st.text("Probability (0: dumper dent, 1: bumper scratch, 2: door dent, 3: door scratch, 4: glass shatter, 5: head lamp, 6: multiple damage, 7: tail lamp")
    st.write(prediction)