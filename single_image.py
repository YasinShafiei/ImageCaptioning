"""
Captioning single images using the model 
----------------------------------------
By Yasin Shafiei
April 9, 2025
"""
from model import EncoderCNN, DecoderLSTM
from train import EMBED_DIM, HIDDEN_SIZE, IMAGE_SIZE, device
import pickle 
import torchvision.transforms.transforms as transforms
import torch
from PIL import Image
import matplotlib.pyplot as plt
from utils import ENCODER_WEIGHT_DIR, DECODER_WEIGHT_DIR

# loading STOI 
with open('stoi.pkl', 'rb') as f:
    stoi = pickle.load(f)

# define vocab size
vocab_size = len(stoi) 

# Define the encoder and decocer
encoder = EncoderCNN(EMBED_DIM).to(device)
decoder = DecoderLSTM(EMBED_DIM, HIDDEN_SIZE, vocab_size).to(device)

# load model weights 
encoder.load_state_dict(torch.load(ENCODER_WEIGHT_DIR))
decoder.load_state_dict(torch.load(DECODER_WEIGHT_DIR))

print("MODEL WEIGHTS LOADED")

def generate_caption(image_path):

    # set model to evaluation mode
    encoder.eval()
    decoder.eval()

    # load image
    image = Image.open(image_path)
    # apply transforms 
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device) 

    # pass the image through encoder
    features = encoder(image_tensor)
    output_ids = decoder.sample(features, stoi)

    itos = {i:s for s, i in stoi.items()}

    words = [itos[idx] for idx in output_ids if idx != stoi['<EOS>']]

    plt.imshow(image)
    plt.axis('off')
    plt.title(' '.join(words), loc='center')
    plt.show()
    # return ' '.join(words)


if __name__ == "__main__":
    generate_caption("/dataset/Images/529198549_5cd9fedf3f.jpg")
    