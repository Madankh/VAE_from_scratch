import torch
import torchvision.datasets as datasets
from tqdm import tqdm
import torch.nn as nn
from model import VariationalAutoEncoder
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 28*28
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 10000
BATCH_SIZE = 32
LR_RATE = 3e-4 # learning rate

# Dataset loading
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR_RATE)
loss_fn = torch.nn.BCELoss(reduction='sum')

# Training loop
for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (x, _) in loop:
        x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
        x_reconstructed, mu, sigma = model(x)
        
        # Compute loss
        reconstruction_loss = loss_fn(x_reconstructed, x)
        kl_div = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        loss = reconstruction_loss + kl_div
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        loop.set_postfix(loss=loss.item())
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item()}")