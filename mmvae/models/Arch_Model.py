import torch
import torch.nn as nn
import mmvae.models.utils as utils
import mmvae.models as M

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        #Encoder
        self.encoder = nn.Sequential(
            nn.Linear(60664, 3072),
            nn.ReLU(),
            nn.BatchNorm1d(3072, 0.8),
            nn.Linear(3072, 900),
            nn.ReLU(),
            nn.BatchNorm1d(900, 0.8),
            nn.Linear(900, 300),
            nn.ReLU(),
            nn.BatchNorm1d(300, 0.8)
        )
        
        self.fc_mu = nn.Linear(300, 128)
        self.fc_var = nn.Linear(300, 128)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 300),
            nn.ReLU(),
            nn.BatchNorm1d(300, 0.8),
            nn.Linear(300, 900),
            nn.ReLU(),
            nn.BatchNorm1d(900, 0.8),
            nn.Linear(900, 3072),
            nn.ReLU(),
            nn.BatchNorm1d(3072, 0.8),
            nn.Linear(3072, 60664),
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
    
