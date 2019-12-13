import torch
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, max_num_node, max_prev_node):
        super(VAE, self).__init__()

        self.N = max_num_node
        self.M = max_prev_node
        self.fc1 = nn.Linear(self.N*self.M, self.N*self.M)
        self.fc21 = nn.Linear(self.N*self.M, self.N*self.M)
        self.fc22 = nn.Linear(self.N*self.M, self.N*self.M)
        self.fc3 = nn.Linear(self.N*self.M, self.N*self.M)
        self.fc4 = nn.Linear(self.N*self.M, self.N*self.M)

        #flow model
        ###########################################################################################
        # self.prior = AFPrior(hidden_size=self.M, zsize=self.N, dropout_p=0, dropout_locations=['prior_rnn'], prior_type='AF', num_flow_layers=1, rnn_layers=2,
        #                       transform_function='nlsq', hiddenflow_params=hiddenflow_params)
        ###########################################################################################

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def inference(self, x):
        mu, logvar = self.encode(x.view(-1, self.N*self.M))
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):
        z, mu, logvar = self.inference(x)
        return self.decode(z), mu, logvar
