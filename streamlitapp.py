import streamlit as st
from src.detect_glaucoma.utils import detect_glaucoma
# from src.detect_glaucoma.glaucoma import prediction
from src.detect_glaucoma.utils import  detect_glaucoma
import cv2
from src.detect_glaucoma.utils import detect_glaucoma
import traceback
from PIL import Image




#Creating title for the appimage_path
st.title("GLAUCOMA DETECTION APP")


#Create  a form using st.form
with st.form("user inputs"):
    #File Upload


    
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
   

    #Add Button
    button = st.form_submit_button("Predict Glaucoma")


    # Check if the button is clicked and all fields have input

    if button and uploaded_file  is not None:
        with st.spinner("loading..."):
            try:
            #    image = custom_preprocessing(uploaded_file)
               # Detect glaucoma in the image
               prediction = detect_glaucoma(uploaded_file)

               if prediction is not None:
                   print("Glaucoma detection result:", prediction)
               else:
                   print("Error occurred while processing the image.")
            
               response = prediction
               st.write(response)
                

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error")
            
         