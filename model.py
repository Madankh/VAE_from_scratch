import torch
import torch.nn as nn
import torch.functional as F

# Input img -> Hidden dim -> mean, std -> reparameterization trick -> Decoder ->Output img
class VariationalAutoEncoder(nn.Module):
    def __init__(self , input_dim, h_dim=200, z_dim=20):
        super().__init__()
        # Encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        # Decoder 
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()
    def encoder(self , x):
        # q_phi(z|x)
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h) , self.hid_2sigma(h)
        return mu, sigma 

    def decoder(self, z):
        # p_theta(x|z)
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))

    def forward(self, x):
        mu , sigmoid = self.encoder(x)
        epslion = torch.rand_like(sigmoid)
        z_reparametrized = mu + sigmoid * epslion
        x_reconstructed = self.decoder(z_reparametrized)
        return x_reconstructed, mu, sigmoid

if __name__ == "__main__":
    x = torch.randn(4,28*28)
    vae = VariationalAutoEncoder(input_dim=784)
    x_reconstructed, mu, sigmoid = vae(x)
    print(mu.shape)
    print(sigmoid.shape)
    print(x_reconstructed.shape)