import re

ENCODER_WEIGHT_DIR = 'Image captioning/outputs/encoder.pt'
DECODER_WEIGHT_DIR = 'Image captioning/outputs/decoder.pt'

def tokenize_spacy(text, nlp):
    """
    Tokenizing using Spacy 
    """
    doc = nlp(text)

    return [token.text for token in doc if not token.is_space]

def cleaning_text(text):
    """
    Cleans and preprocesses a given text by converting it to lowercase, removing HTML tags, URLs, 
    """
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)               
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r"[^a-z0-9\s']", '', text)          
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text