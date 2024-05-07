import torch
import torch.nn as nn
import torch.functional as F

class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encoder(self , x):
        # q_phi(z|x)
        pass 

    def decoder(self, z):
        # p_theta(x|z)
        pass 

    def forward(self, x):
        pass

if __name__ == "__main__";
    x = torch.randn(1, 784)
    vae = VariationalAutoEncoder()
    print(vae(x).shape)