from detectobject import detect_object
import streamlit as st
import numpy as np
from PIL import Image

def upload_image():
    uploaded_image=st.file_uploader('please upload valid image',type=['png','jpg','jpeg'])
    if uploaded_image is not None:
        try:
            image=Image.open(uploaded_image)
        except Exception:
            st.error('Error : Invalid Image')
            
        else:
            img_array=np.array(image)
            return img_array
    
    
def main():
    st.title('Object Detection ')
    img_array=upload_image()
    
    if isinstance(img_array,np.ndarray):
        image=detect_object(img_array)
        st.image(image)
        
    
if __name__=='__main__':
    main()
