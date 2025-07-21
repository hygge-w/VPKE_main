import json
import torch
from PIL import Image
import os
from model.clip_model import load_clip
clip_model = load_clip(config.clip_arch)



def preprocess_text_features(b_json, clip_model, tokenizer, device):
    """Preprocess all text features from b_json."""
    text_features = {}
    for element in b_json:
        original_element = element
        if tokenizer is not None:
            element = tokenizer(element, return_tensors='pt', padding=True, truncation=True)

        if isinstance(element, torch.Tensor):
            element = element.to(device)
        else:
            element = {key: val.to(device) for key, val in element.items()}

        with torch.no_grad():
            text_features[original_element] = clip_model.encode_text(element)
    return text_features


def process_sentence(sentence, text_features, b_json, clip_model, tokenizer, device):
    """Process a single sentence to find top 10 least similar elements."""
    similarity_scores = {element: 0 for element in b_json}
    sentence_text = sentence['raw']

    # Convert sentence to feature representation
    if tokenizer is not None:
        sentence_text = tokenizer(sentence_text, return_tensors='pt', padding=True, truncation=True)

    if isinstance(sentence_text, torch.Tensor):
        sentence_text = sentence_text.to(device)
    else:
        sentence_text = {key: val.to(device) for key, val in sentence_text.items()}

    sentence_features = clip_model.encode_text(sentence_text)

    # Calculate similarity scores
    for element in b_json:
        element_features = text_features[element]
        with torch.no_grad():
            similarity = cosine_similarity_loss(sentence_features, element_features)
        similarity_scores[element] = similarity.item()

    # Get top 10 least similar elements
    sorted_elements = sorted(similarity_scores.keys(),
                             key=lambda x: similarity_scores[x],
                             reverse=False)
    return sorted_elements[:10]  # Select the 10 most relevant key entities


def main():
    # Load data files
    with open('/database/blip#f30k_train_words.json', 'r') as file:
        b_json = json.load(file)

    with open('/database/dataset_flickr30k.json', 'r') as f:
        data = json.load(f)

    # Initialize CLIP model and device (assuming these are defined elsewhere)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Just choose one of them
    from transformers import CLIPTokenizer 
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16", TOKENIZERS_PARALLELISM=False)
    from modules.tokenization_clip import SimpleTokenizer
    tokenizer = SimpleTokenizer()
    clip_model = clip_model  # Replace with your CLIP model
    tokenizer = tokenizer   # Replace with your tokenizer

    # Preprocess all text features
    text_features = preprocess_text_features(b_json, clip_model, tokenizer, device)

    # Process each image and sentence
    for image in data['images']:
        for sentence in image['sentences']:
            top_10_elements = process_sentence(sentence, text_features, b_json,
                                               clip_model, tokenizer, device)

            # Append the top 10 elements to the sentence
            sentence['raw'] += " .#" + "#".join(top_10_elements)

    # Save the modified data
    with open('blip#dataset_flickr30k_wenbenxiangsidu.json', 'w') as f:
        json.dump(data, f, indent=None)


if __name__ == "__main__":
    main()
