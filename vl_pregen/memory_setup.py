import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
import string
import json
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

def extract_key_info(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)

    phrases = []
    current_phrase = []
    for word, pos in tagged_tokens:
        if pos.startswith('NN') or pos.startswith('JJ') or pos.startswith('NNS') or pos.startswith('NNP') or pos.startswith('NNPS'):
            current_phrase.append(word)
        else:
            if current_phrase:
                phrases.append(' '.join(current_phrase))
                current_phrase = []
    if current_phrase:
        phrases.append(' '.join(current_phrase))

    return phrases

with open('/database/blip#coco_dataset_flickr30k#yuanju.json', 'r') as f:
    data = json.load(f)

# Count images by split
train_count = 0
test_count = 0
val_count = 0

image_count = len(data['images'])

for image in data['images']:
    if image['split'] == 'train':
        train_count += 1
    elif image['split'] == 'testall':
        test_count += 1
    elif image['split'] == 'val':
        val_count += 1

unique_tokens = set()
i = 0
for image in data['images']:
    if image['split'] == 'train': #Training data only
        i += 1
        raw_sentences = [sentence['raw'] for sentence in image['sentences']]
        sentence = raw_sentences[1]
        sentence = sentence.split('|')[1].strip()
        phrases = extract_key_info(sentence)
        for phrase in phrases:
            if len(word_tokenize(phrase.lower())) > 1:
                unique_tokens.add(phrase)
            else:
                tokens = word_tokenize(phrase.lower())
                filtered_tokens = [word for word in tokens if
                                 word not in stopwords.words('english') and word not in string.punctuation]
                unique_tokens.update(filtered_tokens)

unique_words_list = list(unique_tokens)

with open('blip#f30k_train_words.json', 'w') as f:
    json.dump(unique_words_list, f, indent=None)
