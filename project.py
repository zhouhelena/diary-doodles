import os
import nltk
import torch
from PIL import Image
from google.cloud import vision
from diffusers import AutoPipelineForText2Image
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.language_models import TextGenerationModel


# Set GPU
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("running on device {device}")
    if device == "cuda":
        print(torch.backends.cudnn.enabled)
        print(torch.cuda.is_available())
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(0))
    else:
        print("Running on cpu")
except Exception as e:
        print(f"An error occurred: {e}")
finally:
    torch.cuda.empty_cache()

# Set Google Cloud credentials and project
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google-vision-credentials.json'  
vertexai.init(project='master-pager-420817')
gemini_model = GenerativeModel(model_name="gemini-1.0-pro-vision-001")

# Initialize Google Cloud Storage
storage_client = storage.Client()
bucket_name = 'diary-doodles'
bucket = storage_client.bucket(bucket_name)

# Function to upload image and detect text using Google Cloud Vision
def detect_text(image_data):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_data)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description if texts else ""

# Function to summarize diary text using Vertex AI
def summarize_diary(diary_text):
    model = TextGenerationModel.from_pretrained("text-bison@002")
    response = model.predict(
        f"""Diary Text: {diary_text}
        
        Q: Summarize each verb and the object behind it. For example, 'Ate breakfast. Went to computer science class. Went to the gym. Went to sleep.'. 
        """,
        temperature=0.5, # Randomness in generation [0.2, 1]
        max_output_tokens=256,
        top_p=0.9, # Probability threshold [0, 1]
        top_k=40 # Top-k sampling
    )
    return response.text

# Function to upload image to Google Cloud Storage
def upload_image_to_gcs(image_data, destination_blob_name):
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(image_data, content_type='image/jpeg')
    return f"gs://{bucket_name}/{destination_blob_name}"

# Function to generate image description using Gemini model
def caption_image(image_data, image_filename):
    image_uri = upload_image_to_gcs(image_data, image_filename)
    response = gemini_model.generate_content(
        [
            Part.from_uri(image_uri, mime_type="image/jpeg"),
            "describe the person's appearance only. For example, 'a girl with big blue eyes, sphere shape face, and blue skin'.",
        ]
    )
    print(response)
    return response.text

# Function to generate image from text prompt
def generate_image_from_text(prompt, num_inference_steps=50, guidance_scale=10.0): # 1. [20, 100]  2. [7.5, 15]
    formatted_prompt = f"Create a detailed cartoon image. {prompt}. The face should be clearly visible, and the proportions should be accurate."
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe = pipe.to(device)
    result = pipe(prompt=formatted_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    return result.images[0]


# Main function
def main():
    # Load images
    diary_image_path = input("Enter the path to your diary image: ")
    person_image_path = input("Enter the path to your image: ")

    with open(diary_image_path, 'rb') as f:
        diary_image_data = f.read()

    with open(person_image_path, 'rb') as f:
        person_image_data = f.read()

    # Generate description for person image
    print('Analyzing person...')
    person_description = caption_image(person_image_data, 'person.jpg').lower()
    print('Person description:')
    print(person_description)

    # Extract and summarize diary text
    print('Extracting text...')
    extracted_text = detect_text(diary_image_data)
    print('Extracted text:')
    print(extracted_text)

    print('Analyzing text and splitting into scenes...')
    summary = summarize_diary(extracted_text)
    print(summary)
    scenes = nltk.tokenize.sent_tokenize(summary)

    # Generate image for each scene
    for i, scene in enumerate(scenes):
        scene_prompt = f"Depict {person_description} in this scene in a cartoon style: {scene}"
        print(f'Scene {i+1}: {scene_prompt}')
        print('Generating image for this scene...')
        generated_image = generate_image_from_text(scene_prompt)
        output_image_path = f'scene_{i+1}.png'
        generated_image.save(output_image_path)
        print(f'Scene {i+1} image saved as {output_image_path}')

if __name__ == '__main__':
    main()