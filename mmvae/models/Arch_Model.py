import torch
import torch.nn as nn
import mmvae.models.utils as utils
import mmvae.models as M

# neuron_sizes  = {
#     "xlarge": [8192, 2048, 512],    # Very large configuration
#     "large": [4096, 1200, 400],     # Existing large configuration
#     "upper-medium": [3072, 900, 300],  # Intermediate between large and medium
#     "medium": [2048, 600, 200],     # Existing medium configuration
#     "lower-medium": [1536, 450, 150],  # Intermediate between medium and small
#     "small": [1024, 300, 100],      # Existing small configuration
#     "xsmall": [512, 150, 50]        # Smaller than the existing small configuration
# }
neuron_sizes = [8192, 2048, 512]

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
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
        
        # These layers connect to the latent space dimensions
        self.fc_mu = nn.Linear(neuron_sizes[2], 128)  # Keeping the latent space dimension fixed at 128
        self.fc_var = nn.Linear(neuron_sizes[2], 128)
        
        # Decoder
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
    
