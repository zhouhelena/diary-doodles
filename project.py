import os
import nltk
import torch
import streamlit as st
from google.cloud import vision
from diffusers import AutoPipelineForText2Image
import torch
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.language_models import TextGenerationModel
from google.cloud import storage
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# TODO: Need device to be GPU
device = "cuda" 
print(torch.backends.cudnn.enabled)
print(torch.cuda.is_available())  
print(torch.cuda.device_count())  
print(torch.cuda.get_device_name(0))
torch.cuda.empty_cache()

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google-vision-credentials.json'
vertexai.init(project='master-pager-420817')
gemini_model = GenerativeModel(model_name="gemini-1.0-pro-vision-001")

storage_client = storage.Client()
bucket_name = 'diary-doodles'
bucket = storage_client.bucket(bucket_name)

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def load_image_uploader(label):
    uploaded_file = st.file_uploader(label=label)
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(uploaded_file)
        return image_data
    return None

def detect_text(image_data):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_data)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description if texts else ""

def summarize_diary(diary_text):
    model = TextGenerationModel.from_pretrained("text-bison@002")
    response = model.predict(
        f"""Diary Text: {diary_text}
        
        Q: Summarize the diary into exactly 4 sentences that capture the main events or themes in order. Each sentence must be under 5 words and focus exactly 1 action. For example, 'Eating breakfast. Going to computer science class. Going to the gym. Going to sleep.'. 
        """,
        temperature=0.5,
        max_output_tokens=256,
        top_p=0.9,
        top_k=40
    )
    return response.text


def upload_image_to_gcs(image_data, destination_blob_name):
    """Uploads a file to the bucket."""
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(image_data, content_type='image/jpeg')
    return f"gs://{bucket_name}/{destination_blob_name}"

def caption_image(image_data, image_filename):
    image_uri = upload_image_to_gcs(image_data, image_filename)

    gemini_model = GenerativeModel(model_name="gemini-1.0-pro-vision-001")
    response = gemini_model.generate_content(
        [
            Part.from_uri(image_uri, mime_type="image/jpeg"),
            "In under ten words total, describe the person's appearance and clothing. Ignore any objects the person may be wearing or carrying, and ignore the location. For example, 'a girl with black hair wearing a white dress'.",
        ]
    )
    print(response)
    return response.text

def generate_image_from_text(prompt):
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe = pipe.to("cuda")

    result = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0)
    return result.images[0]

def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_sentence = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_sentence)
    
def main():
    st.title('Diary Doodles')
    
    diary_image = load_image_uploader('Select an image of your diary')
    person_image = load_image_uploader('Select an image of yourself')

    if st.button('Turn into a picture book'):
        # Process person image
        if person_image is not None:
            with st.spinner('Analyzing person...'):
                person_description = caption_image(person_image, 'diary.jpg').lower()

        # Process diary image
        if diary_image is not None:
            with st.spinner('Extracting text...'):
                extracted_text = detect_text(diary_image)

            with st.spinner('Analyzing text...'):
                summary = summarize_diary(extracted_text)

            scenes = nltk.tokenize.sent_tokenize(summary)
            cols = st.columns(2)
            for i, scene in enumerate(scenes):
                with cols[i % 2]:
                    scene_prompt = f"In cute cartoon picture book style: {person_description} {scene}"
                    scene_prompt_cleaned = remove_stopwords(scene_prompt)
                    with st.spinner('Generating image...'):
                        generated_image = generate_image_from_text(scene_prompt_cleaned)
                        st.image(generated_image, caption=scene)
        else:
            st.write('No image selected.')

if __name__ == '__main__':
    main()
