import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

import torchvision.datasets as datasets

dataset_path="~/datasets"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dim = 784 # each image is 28*28 = 784 pixels
batch_size = 100
hidden_dim = 400
latent_dim = 200
learning_rate = 1e-3
epochs = 30

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# MNIST
mnist_transform = transforms.Compose([
    transforms.ToTensor()
])

kwargs = {'num_workers':1, 'pin_memory':True}
train_dataset = MNIST(dataset_path, transform=mnist_transform,train=True, download=True)
test_dataset = MNIST(dataset_path, transform=mnist_transform, train=False, download=False)

train_loader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=True,**kwargs)
test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size,shuffle=True , **kwargs)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim , latent_dim):
        super(Encoder, self).__init__()
        # Hidden layers used to process the inputs before converting to latents
        self.hidden_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(0.2),
            nn.Linear(hidden_dim , hidden_dim),
            nn.LeakyReLU(0.2)
        )
        # Laten representations encoded into mean and lo variance vector
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_variance = nn.Linear(hidden_dim, latent_dim)
        self.training = True
    def forward(self, x):
        hidden = self.hidden_layer(x)
        mean = self.mean(hidden)
        log_variance = self.log_variance(hidden)
        return mean, log_variance
    

class Decoder(nn.Module):
    def __init__(self , latent_dim, hidden_dim, output_dim):
        super(Decoder , self).__init__()
        
        self.hidden_layer = nn.Sequential(
            nn.Linear(latent_dim , hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim , hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim , output_dim)
        )
    def forward(self, x):
        hidden = self.hidden_layer(x)
        x_hat = torch.sigmoid(hidden)
        return x_hat
    