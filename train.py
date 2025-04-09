"""
Training and saving the image captioning model
----------------------------------------------
By Yasin Shafiei
April 7, 2025
"""
import torch
import torch.nn as nn
import torch.optim.adam
from torch.utils.data import DataLoader 
from torchvision.transforms import transforms
from model import EncoderCNN, DecoderLSTM
from data import IC2Dataset, load_data, collate_fn
import time
import os
import pickle

NUM_EPOCHS = 40
EMBED_DIM = 512 
HIDDEN_SIZE = 512
BATCH_SIZE = 16
IMAGE_SIZE = 224
LR = 1e-3
WEIGHTS_DIR = 'outputs/'
DATASET_PATH = 'dataset/'
device = 'mps'

def main():
    """
    Loading the data and model, training the model on Flicker8K dataset, saving the model and weights. 
    """
    ### Loading the dataset 
    image_paths, encoded_captions, stoi, itos = load_data(DATASET_PATH)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    dataset = IC2Dataset(image_paths, encoded_captions, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # define encoder and decoder
    encoder = EncoderCNN(EMBED_DIM).to(device)
    decoder = DecoderLSTM(EMBED_DIM, HIDDEN_SIZE, len(stoi)).to(device)

    parameters = list(encoder.parameters()) + list(decoder.parameters())

    # define loss function and optimizer 
    loss_function = nn.CrossEntropyLoss(ignore_index=0) # don't punish padding tokens
    optimizer = torch.optim.Adam(parameters, lr=LR)

    print("TRAINING OF IMAGE CAPTIONING MODEL")
    print("=========================================")
    print(f"Number of model parameters: {sum(p.numel() for p in parameters)}")
    print(f"Number of Epochs: {NUM_EPOCHS}")
    print(f"Using: {device}")
    print("=========================================")
 
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0 
        e_i = 0
        total_tokens = 0
        start_time = time.time()
        
        for i, (images, captions) in enumerate(dataloader):
            images = images.to(device)
            captions = captions.to(device)
            
            # Count total tokens in this batch (excluding padding)
            total_tokens += (captions != 0).sum().item()
            
            # prepare the inputs and targets
            # For the inputs we remove the last token from the caption which is <EOS>
            # And for the targets, we remove the first token which is <SOS>.
            # The inputs will be used for the decoder
            # The targets will be used for calculating the loss
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            # run through the model 
            features = encoder(images)
            outputs = decoder(features, inputs)

            # reshape for calculating the loss
            outputs = outputs.reshape(-1, outputs.size(2)) # (B * T, voca_size)
            targets = targets.reshape(-1) # (B * vocab_size)

            loss = loss_function(outputs, targets)

            # Backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            e_i += 1

        epoch_time = time.time() - start_time
        tokens_per_second = total_tokens / epoch_time
        
        print(f'Epoch: {epoch+1}/{NUM_EPOCHS} ||| '
              f'Loss: {epoch_loss/len(dataloader):.4f} ||| '
              f'Time: {epoch_time:.2f}s ||| '
              f'Speed: {tokens_per_second:.2f} tokens/s')
        
    torch.save(encoder.state_dict(), os.path.join(WEIGHTS_DIR, 'encoder.pt'))
    print(f'Encoder weights saved at {WEIGHTS_DIR}') 
    torch.save(decoder.state_dict(), os.path.join(WEIGHTS_DIR, 'decoder.pt'))
    print(f'Decoder weights saved at {WEIGHTS_DIR}')

    with open('stoi.pkl', 'wb') as f:
        pickle.dump(stoi, f)

if __name__ == "__main__":
    main()