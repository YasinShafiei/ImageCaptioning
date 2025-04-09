"""
Data loading class for image captioning 
---------------------------------------
By Yasin Shafiei
April 6, 2025
"""
import os 
import torch
import spacy 
from PIL import Image
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

from utils import tokenize_spacy, cleaning_text

nlp = spacy.load('en_core_web_sm')

class IC2Dataset(Dataset):
    """
    Custom Dataset class for Image Captioning.
    This class is used to load image-caption pairs for training or evaluation.
    """

    def __init__(self, image_paths, encoded_captions, transform):
        super(IC2Dataset, self).__init__()

        self.image_paths = image_paths  
        self.encoded_captions = encoded_captions 
        self.transform = transform 

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Retrieves the image and caption at the specified index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the transformed image and the encoded caption.
        """
        # Load and transform the image at the given index.
        image = self.transform(Image.open(self.image_paths[index]))
        # Retrieve the encoded caption corresponding to the image.
        caption = torch.tensor(self.encoded_captions[index])

        return image, caption

def load_data(raw_path):
    """
    Loading images paths, preprocessing the captions
    Args: 
        raw_path: Path to where the txt file and images folder are located
    Returns:
        image_dirs: A list containing directory of all images 
        encoded_captions: Encoded captions 
        stoi: character-to-index lookup table 
        itos: index-to-character lookup table
    """
    # load raw caption and image paths
    txt_file = open(f'{raw_path}/captions.txt').read().splitlines()
    images_dirs = [os.path.join(raw_path, 'Images', data.split(',')[0]) for data in txt_file[1:]]
    captions = [data.split(',')[1] for data in txt_file[1:]]

    ### preprocess captions 
    captions = [' '.join(c.split()[:-1]) for c in captions] 
    tokenized = [tokenize_spacy(cleaning_text(text), nlp) for text in captions]
    # define stoi and itos 
    all_tokens = [token for sentence in tokenized for token in sentence]
    vocab = Counter(all_tokens)
    stoi = {word: i+3 for i, (word, _) in enumerate(vocab.most_common())}
    stoi['<PAD>'] = 0
    stoi['<SOS>'] = 1 
    stoi['<EOS>'] = 2
    itos = {i: word for word, i in stoi.items()}

    encoded_captions = [[stoi[token] for token in sentence] for sentence in tokenized]

    return images_dirs, encoded_captions, stoi, itos 

def collate_fn(batch):
    """
    Custom collate function for preparing batches of image-caption pairs.
    Args:
        batch (list of tuples): A batch of data where each element is a tuple 
            containing an image tensor and its corresponding caption tensor.
    Returns:
        tuple: A tuple containing:
            - images (torch.Tensor): A tensor of stacked image tensors from the batch.
            - padded_captions (torch.Tensor): A tensor of caption tensors padded to the 
              same length, with padding value 0.
    """
    # get the images and captions from the batch
    images, captions = zip(*batch)
    # put all images together 
    images = torch.stack(images)
    # add <EOS> and <SOS> tokens 
    captions = [
        torch.tensor([1] + c.tolist() + [2])
        for c in captions
    ]  
    # perform padding 
    padded_captions = pad_sequence(captions, batch_first=True, padding_value=0)

    return images, padded_captions

### FOR TESTING
if __name__ == "__main__":
    path = 'dataset/'
    images_path, encoded_captions, stoi, itos = load_data(path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    dataset = IC2Dataset(images_path, encoded_captions, transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    for i, (image, caption) in enumerate(dataloader):
        if i == 0:
            print(image.shape)
            print(caption.shape)
            break
