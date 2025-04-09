"""
Encoder and Decoder for image captioning
----------------------------------------
By Yasin Shafiei
April 5, 2025
"""
import warnings
warnings.filterwarnings('ignore')

import torch 
import torch.nn as nn
import torch.nn.init as init
from torchvision.models import resnet101 

class EncoderCNN(nn.Module):
    """
    EncoderCNN is a convolutional neural network (CNN) encoder designed to extract feature embeddings from input images. 
    It utilizes a pre-trained ResNet-101 model as the backbone for feature extraction, followed by a fully connected 
    layer to transform the extracted features into a specified embedding dimension.
    Attributes:
        resnet (nn.Sequential): A modified ResNet-101 model with the classification head removed, used for feature extraction.
        fc (nn.Linear): A fully connected layer that maps the extracted features to the desired embedding dimension.
        relu (nn.ReLU): A ReLU activation function applied after the fully connected layer.
    Methods:
        __init__(embed_dim):
            Initializes the EncoderCNN with the specified embedding dimension, loads the pre-trained ResNet-101 model, 
            removes its classification head, freezes its parameters, and defines the final fully connected layer.
        forward(images):
            Performs a forward pass through the encoder, extracting features from the input images and transforming 
            them into the specified embedding dimension.
    """
    def __init__(self, embed_dim):
        super(EncoderCNN, self).__init__()

        # load the pretrained ResNet model 
        resnet = resnet101(pretrained=True)
        # remove the classification head 
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        # freeze the parameters
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # define a final classification layer for transforming the features into embedding dimensions
        self.fc = nn.Linear(resnet.fc.in_features, embed_dim)
        self.relu = nn.ReLU()

        # initializig the weight for the fc 
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, images):
        """
        Forward pass through the encoder
        Args:
            images: Input images
        Returns:
            features: Encoded features
        """
        # pass the images through the resnet
        features = self.resnet(images)
        features = features.view(features.size(0), -1)

        # pass through the final fully-connected
        features = self.relu(self.fc(features))

        return features
    
class DecoderLSTM(nn.Module):
    """
    DecoderLSTM is a PyTorch module that implements the decoder part of an image captioning model using an LSTM (Long Short-Term Memory) network. 
    This class is responsible for generating captions for images by taking image feature vectors as input and producing sequences of words as output.
    Attributes:
        embedding (nn.Embedding): Embedding layer to convert word indices into dense vectors of a fixed size.
        lstm (nn.LSTM): LSTM layer for sequential modeling, which processes the embedded word vectors.
        linear (nn.Linear): Fully connected layer to map the LSTM outputs to vocabulary size for classification.
        init_h (nn.Linear): Linear layer to initialize the hidden state of the LSTM using image features.
        init_c (nn.Linear): Linear layer to initialize the cell state of the LSTM using image features.
    Methods:
        forward(features, captions):
            Performs a forward pass through the decoder, generating logits for each word in the caption sequence.
        sample(features, stoi, max_len=30, device='mps'):
            Generates a caption for a given image by sampling words sequentially from the decoder.
    """

    def __init__(self, embed_dim, hidden_size, vocab_size, num_layers=2, lstm_dropout=0.2):
        super(DecoderLSTM, self).__init__()
        
        self.num_layers = num_layers 

        # define the embedding 
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # define the LSTM layer 
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout)

        # define the final classification layer 
        self.linear = nn.Linear(hidden_size, vocab_size)

        # define two linear layers for making the image compatible with the hidden and cell states 
        # in the decoder, we pass the image to be the initial hidden and cell states 
        self.init_h = nn.Linear(embed_dim, hidden_size)
        self.init_c = nn.Linear(embed_dim, hidden_size)

        # initialize weights 
        self._init_weights()

    def forward(self, features, captions):
        """
        Forward pass through the decoder
        Args:
            features: Image feature vectors, which is the result of passing the image through encoder 
            capions: Image captions in encoded format
        Returns:
            logits: Logits are the raw, unnormalized scores output by a model for each possible option
        """
        # get embed
        embed = self.embedding(captions)

        # set the initial hidden and cell states to the image (features)
        h0 = self.init_h(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = self.init_c(features).unsqueeze(0).repeat(self.num_layers, 1, 1)

        # pass input to LSTM
        output, _ = self.lstm(embed, (h0, c0))
        logits = self.linear(output)

        return logits
    
    def sample(self, features, stoi, max_len=30, device='mps'):
        """
        Getting samples from the decoder
        Args: 
            features: Image feature vectors, which is the result of passing the image into the decoder
            stoi: Lookup table for converting strings to indexes 
            max_len: Maximum caption length (set to 30 by default)
            device: The devine that the model will use to make predictions
        Return:
            sampled_ids: A list containing numerical values generated by the decoder. Each numerical values represents a character
        """
        sampled_ids = []

        # input token is set to <SOS> at the begining. 
        # the same happens during training
        input_token = torch.tensor([[stoi['<SOS>']]]).to(device)

        # initialize the hidden and cell states 
        h = self.init_h(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = self.init_c(features).unsqueeze(0).repeat(self.num_layers, 1, 1)

        for i in range(max_len):
            embedded = self.embedding(input_token)

            # make prediction 
            output, (h, c) = self.lstm(embedded, (h, c))
            logits = self.linear(output.squeeze(1))

            # convert logits into sample id 
            predicted_id = logits.argmax(1).item() 
            sampled_ids.append(predicted_id)

            # change the input to the predicted id
            input_token = torch.tensor([[predicted_id]]).to(device)

            # break if the predicted token is the end token (<EOS>)
            if predicted_id == stoi['<EOS>']:
                break

        return sampled_ids
    
    def _init_weights(self):
        """
        Initializing the weights of the LSTM
        """
        for name, param in self.lstm.named_parameters():
            # input weight
            if 'weight_ih' in name:
                init.xavier_normal_(param.data)
            # hidden weight 
            if 'weight_hh' in name:
                init.orthogonal_(param.data)
            # bias
            if 'bias' in name:
                init.constant_(param.data, 0)

        ### initialize linear weights
        init.xavier_normal_(self.linear.weight)
        init.constant_(self.linear.bias, 0)
    
### FOR TESTING
if __name__ == "__main__":
    tensor = torch.randn((1, 3, 64, 64))
    captions = torch.randint(1, 20, (1, 10))
    encoder = EncoderCNN(embed_dim=16)
    decoder = DecoderLSTM(16, 16, 20)
    enc_out = encoder(tensor)
    dec_out = decoder(enc_out, captions)
    print(enc_out.shape)
    print(dec_out.shape)
