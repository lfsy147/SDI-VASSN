import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys

use_cuda = torch.cuda.is_available()

arguments = sys.argv
if len(sys.argv) > 1:
        mode = arguments[1]
else:
        mode = "train"
df = pd.read_csv('./stMVC_test_data/DLPFC_151673/stMVC/csn_output.csv')
df = df.drop(df.columns[0], axis=1)
data_values = df.values.T
max_value = np.max(data_values)
data_values=data_values/max_value
data_values=data_values.astype(np.float32)
X_train, X_test = train_test_split(data_values, test_size=0.2, random_state=42)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar, z

input_dim = X_train.shape[1]
hidden_dim = 256
latent_dim = 50
vae = VAE(input_dim, hidden_dim, latent_dim)
if use_cuda:
    vae.cuda()

def loss_function(recon_x, x, mu, logvar):
    bce_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce_loss + kld_loss

optimizer = optim.Adam(vae.parameters(), lr=0.001)
clip_value = 1.0
num_epochs = 30
for epoch in range(num_epochs):
    if mode =="test":
        break

    vae.train()
    train_loss = 0
    for batch_data in tqdm(X_train, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        batch_data = Variable(torch.from_numpy(batch_data)).unsqueeze(0)
        if use_cuda:
            batch_data = batch_data.cuda()
        recon_batch, mu, logvar, _ = vae(batch_data)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, batch_data, mu, logvar)
        loss.backward()
        for group in optimizer.param_groups:
            for param in group['params']:
                param.grad.data = torch.clamp(param.grad.data, -clip_value, clip_value)
        optimizer.step()
        train_loss += loss.item()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss / len(X_train)}')

if mode =="train":

    torch.save(vae.state_dict(), './stMVC_test_data/DLPFC_151673/stMVC/vae_model_github.pth')

loaded_vae = VAE(input_dim, hidden_dim, latent_dim)

loaded_vae.load_state_dict(torch.load('./stMVC_test_data/DLPFC_151673/stMVC/vae_model_github.pth'))
if use_cuda:
    loaded_vae.cuda()

vae.eval()
reconstructed_data = []
for batch_data in data_values:
    batch_data = Variable(torch.from_numpy(batch_data)).unsqueeze(0)
    if use_cuda:
        batch_data = batch_data.cuda()
    output, _, _, z= loaded_vae(batch_data)
    if use_cuda:
        reconstructed_data.append(z.squeeze().cpu().detach().numpy())
    else:
        reconstructed_data.append(z.squeeze().detach().numpy())
df2 = pd.read_csv('./stMVC_test_data/DLPFC_151673/stMVC/gene_output.csv')
reconstructed_data = np.vstack(reconstructed_data)
reconstructed_df = pd.DataFrame(data=reconstructed_data,index=df2.columns[1:])
reconstructed_df.to_csv('./stMVC_test_data/DLPFC_151673/stMVC/reconstructed_gene_data_vae.csv')
