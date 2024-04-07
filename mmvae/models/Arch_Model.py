import torch
import torch.nn as nn
import mmvae.models.utils as utils
import mmvae.models as M

class VAE(nn.Module):
    def __init__(self, neuron_sizes, name) :
        super(VAE, self).__init__()
        print(f'In model init name: {name}, size: {neuron_sizes}')
        #Encoder
        self.encoder = nn.Sequential(
            nn.Linear(60664, neuron_sizes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(neuron_sizes[0], 0.8),
            nn.Linear(neuron_sizes[0], neuron_sizes[1]),
            nn.ReLU(),
            nn.BatchNorm1d(neuron_sizes[1], 0.8),
            nn.Linear(neuron_sizes[1], neuron_sizes[2]),
            nn.ReLU(),
            nn.BatchNorm1d(neuron_sizes[2], 0.8)
        )
        self.fc_mu = nn.Linear(neuron_sizes[2], 128)
        self.fc_var = nn.Linear(neuron_sizes[2], 128)
        self.decoder = nn.Sequential(
            nn.Linear(128, neuron_sizes[2]),
            nn.ReLU(),
            nn.BatchNorm1d(neuron_sizes[2], 0.8),
            nn.Linear(neuron_sizes[2], neuron_sizes[1]),
            nn.ReLU(),
            nn.BatchNorm1d(neuron_sizes[1], 0.8),
            nn.Linear(neuron_sizes[1], neuron_sizes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(neuron_sizes[0], 0.8),
            nn.Linear(neuron_sizes[0], 60664),
            nn.ReLU(),
        )

        utils._submodules_init_weights_xavier_uniform_(self.encoder)
        utils._submodules_init_weights_xavier_uniform_(self.decoder)
        utils._submodules_init_weights_xavier_uniform_(self.fc_mu)
        utils._xavier_uniform_(self.fc_var, -1.0)


    def encode(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_var(x)

    def decode(self, z):
        return self.decoder(z)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
