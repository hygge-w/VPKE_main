import json
import os
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Initialize BLIP-2 model
processor = Blip2Processor.from_pretrained("/home/blip2-opt-6.7b-coco")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Blip2ForConditionalGeneration.from_pretrained(
    "/home/blip2-opt-6.7b-coco",
    torch_dtype=torch.float16
).to(device)


def generate_caption(image_path, model, processor):
    """Generate caption for an image using BLIP-2 model"""
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def extract_phrases(text):
    """Extract noun and adjective phrases from text"""
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)

    phrases = []
    current_phrase = []
    for word, pos in tagged_tokens:
        if pos.startswith(('NN', 'JJ')):  # Combined all noun and adjective tags
            current_phrase.append(word)
        else:
            if current_phrase:
                phrases.append(' '.join(current_phrase))
                current_phrase = []
    if current_phrase:
        phrases.append(' '.join(current_phrase))
    return phrases


def merge_and_filter(caption, generated_text):
    """Merge and filter phrases from generated text with original caption"""
    generated_phrases = extract_phrases(generated_text)
    caption_phrases = extract_phrases(caption)

    # Filter out phrases already in caption and stopwords
    unique_phrases = [
        phrase for phrase in set(generated_phrases) - set(caption_phrases)
        if all(word.lower() not in stopwords.words('english')
               for word in phrase.split())
    ]

    return f"{caption} #{'#'.join(unique_phrases)}".strip()


def process_images(input_json, output_json, image_dir):
    """Process all images in the input JSON file"""
    with open(input_json, 'r') as f:
        data = json.load(f)

    for image in data['images']:
        image_path = os.path.join(image_dir, image["file_name"])
        generated_text = generate_caption(image_path, model, processor)

        for sentence in image['sentences']:
            caption = sentence['raw']
            merged_sentence = merge_and_filter(caption, generated_text)
            sentence['raw'] = f"{merged_sentence} | {generated_text}"

    with open(output_json, 'w') as f:
        json.dump(data, f, indent=None)


if __name__ == "__main__":
    process_images(
        input_json='testall_coco.json',
        output_json='/icislab/volume1/186/W_s/W_S/ESL-main/data2/blip#train_coco#yuanju.json', #Save the auxiliary caption file path
        image_dir='/icislab/volume1/186/W_s/val2014/' #Load the original image data
    )
