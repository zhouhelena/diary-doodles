import requests
import os
import nltk
import torch
from io import BytesIO
from PIL import Image
import streamlit as st
from google.cloud import vision
import tensorflow as tf
from diffusers import AutoPipelineForText2Image
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import numpy as np
from scipy.cluster.vq import kmeans2
from transformers import AutoTokenizer, BitsAndBytesConfig
from LLaVA.llava.model import LlavaLlamaForCausalLM
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from LLaVA.llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

# TODO: Need device to be GPU
device = "cuda" 
print(torch.backends.cudnn.enabled)
print(torch.cuda.is_available())  
print(torch.cuda.device_count())  
print(torch.cuda.get_device_name(0))

model_path = "4bit/llava-v1.5-13b-3GB"
kwargs = {"device_map": "auto"}
kwargs['load_in_4bit'] = True
kwargs['quantization_config'] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
vision_tower.to(device='cuda')
image_processor = vision_tower.image_processor

def load_image_uploader(label):
    uploaded_file = st.file_uploader(label=label)
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(uploaded_file)
        return image_data
    return None

def detect_text(image_data):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google-vision-credentials.json'
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_data)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description if texts else ""

def get_text_embeddings(text):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)

    # Tokenize and encode the text for the model
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Use mean pooling to convert the token embeddings to sentence embeddings
    attention_mask = encoded_input['attention_mask']
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).squeeze().cpu().numpy()

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

def caption_image(image_file, prompt):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    disable_torch_init()
    conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    inp = f"{roles[0]}: {prompt}"
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    raw_prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
      output_ids = model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2,
                                  max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs
    output = outputs.rsplit('</s>', 1)[0]
    return image, output

def generate_image_from_text(prompt):
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
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

        # Process person image
        if person_image is not None:
            st.write('Analyzing person...')
            person_description = caption_image(person_image, "Describe the person's appearance and clothing in extreme detail. Ignore the location.")
            st.write('Person Description:')
            st.write(person_description)

        # Process diary image
        if diary_image is not None:
            st.write('Extracting text...')
            extracted_text = detect_text(diary_image)
            st.write('Extracted Text:')
            st.write(extracted_text)

            st.write('Analyzing text and splitting into scenes...')
            scenes = split_text_by_meaning(extracted_text)

            for i, scene in enumerate(scenes):
                scene_prompt = f"The person is {person_description}. Describe this scene: {scene}"
                st.write(f'Scene {i+1}: {scene_prompt}')
                st.write('Generating image for this scene...')
                generated_image = generate_image_from_text(scene_prompt)
                st.image(generated_image, caption=f'Scene {i+1}')

        else:
            st.write('No image selected.')

if __name__ == '__main__':
    main()
