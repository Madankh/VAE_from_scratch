import torch
import torchvision.datasets as datasets 
# from tqdm import tqdm
import torch.nn as nn
from model import VariationalAutoEncoder
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 28*28
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR_RATE = 3e-4 # learning rate

# Dataset loading
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), LR_RATE)

