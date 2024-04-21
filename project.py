import io
import os
from PIL import Image
import streamlit as st
from google.cloud import vision

def detect_text(image_data):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google-vision-credentials.json'
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_data)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description if texts else ""

def load_image():
    uploaded_file = st.file_uploader(label='Select an image of your diary')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(uploaded_file)
        return image_data
    else:
        return None

def main():
    st.title('Diary Doodles')
    
    image_data = load_image()

    if st.button('Turn into a picture book'):
        if image_data is not None:
            st.write('Loading...')
            extracted_text = detect_text(image_data)
            st.write('Extracted Text:')
            st.write(extracted_text)
        else:
            st.write('No image selected.')

if __name__ == '__main__':
    main()
