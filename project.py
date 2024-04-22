import io
import os
import nltk
from PIL import Image
import streamlit as st
from google.cloud import vision
import tensorflow as tf
from diffusers import AutoPipelineForText2Image
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from scipy.cluster.vq import kmeans2

# TODO: Need device to be GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_text_embeddings(text):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Tokenize and encode the text for the model
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Use mean pooling to convert the token embeddings to sentence embeddings
    attention_mask = encoded_input['attention_mask']
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).squeeze().numpy()

def split_text_by_meaning(text, num_clusters=4):
    # Split text into sentences
    sentences = nltk.tokenize.sent_tokenize(text)
    embeddings = np.array([get_text_embeddings(sentence) for sentence in sentences])

    # Perform k-means clustering
    centroids, labels = kmeans2(embeddings, num_clusters, minit='points')

    # Group sentences by cluster
    clustered_sentences = {i: [] for i in range(num_clusters)}
    for sentence, cluster_id in zip(sentences, labels):
        clustered_sentences[cluster_id].append(sentence)

    return [' '.join(clustered_sentences[i]) for i in range(num_clusters)]

def detect_text(image_data):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google-vision-credentials.json'
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_data)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description if texts else ""

def load_image_uploader(label):
    uploaded_file = st.file_uploader(label=label)
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(uploaded_file)
        return image_data
    return None

def generate_image_from_text(prompt):
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe = pipe.to(device)

    result = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=7.5)
    return result.images[0]
    
def main():
    st.title('Diary Doodles')
    print(device)
    
    diary_image = load_image_uploader('Select an image of your diary')
    person_image = load_image_uploader('Select an image of yourself')

    # TODO: Add image pre-processing steps

    if st.button('Turn into a picture book'):
        if diary_image is not None:
            st.write('Extracting text...')
            extracted_text = detect_text(diary_image)
            st.write('Extracted Text:')
            st.write(extracted_text)

            st.write('Analyzing text and splitting into scenes...')
            scenes = split_text_by_meaning(extracted_text)

            for i, scene in enumerate(scenes):
                st.write(f'Scene {i+1}: {scene}')
                st.write('Generating image for this scene...')
                generated_image = generate_image_from_text(scene)
                st.image(generated_image, caption=f'Scene {i+1}')

        else:
            st.write('No image selected.')

if __name__ == '__main__':
    main()
