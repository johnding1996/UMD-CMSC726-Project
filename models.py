import torch
from torch import nn
from torch.nn import functional as F
from transformations import NLSq, Affine

class VAE(nn.Module):
    def __init__(self, num_max_prev_node, encoder_layers, decoder_layers):
        super(VAE, self).__init__()

        self.input_size = num_max_prev_node #M
        self.rnn1 = torch.nn.GRU(input_size=self.input_size, hidden_size=self.input_size, num_layers=encoder_layers, batch_first=True)
        self.rnn21 = torch.nn.GRU(input_size=self.input_size, hidden_size=self.input_size, num_layers=1, batch_first=True)
        self.rnn22 = torch.nn.GRU(input_size=self.input_size, hidden_size=self.input_size, num_layers=1, batch_first=True)
        self.rnn3 = torch.nn.GRU(input_size=self.input_size, hidden_size=self.input_size, num_layers=decoder_layers, batch_first=True)


        #flow model
        ###########################################################################################
        # self.prior = AFPrior(hidden_size=self.M, zsize=self.N, dropout_p=0, dropout_locations=['prior_rnn'], prior_type='AF', num_flow_layers=1, rnn_layers=2,
        #                       transform_function='nlsq', hiddenflow_params=hiddenflow_params)
        ###########################################################################################

    def encode(self, x): #  (batch, seq_len, input_size)
        output, h_n = self.rnn1(x)
        mu, _ = self.rnn21(output) # (output, h_n)?    (batch, seq_len, input_size)
        logvar, _ = self.rnn22(output) # (output, h_n)?  (batch, seq_len, input_size)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar.view(logvar.shape[0], -1) )
        eps = torch.randn_like(std)
        return mu.view(mu.shape[0], -1) + eps*std

    def decode(self, z):
        output, _ = self.rnn3(z)
        return output

    def inference(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z.view(-1, mu.shape[1], mu.shape[2]), mu, logvar

    def forward(self, x):
        z, mu, logvar = self.inference(x)
        return self.decode(z), z, mu, logvar

class FeedForwardNet(nn.Module):

    def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim):
        super(FeedForwardNet, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim),
                  nn.ReLU()]
        for i in range(num_hidden_layers-1) :
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SCFLayer(nn.Module) :
    def __init__(self, num_hidden_layers, hidden_dim, data_dim, conditional_inp_dim, reverse, transform_function=NLSq):
        super(SCFLayer, self).__init__()
        self.data_dim = data_dim

        indices = torch.arange(self.data_dim)
        if reverse :
            self.first_indices = indices[self.data_dim//2:]
            self.second_indices = indices[:self.data_dim//2]
        else :
            self.first_indices = indices[:self.data_dim//2]
            self.second_indices = indices[self.data_dim//2:]

        self.net = FeedForwardNet(num_hidden_layers=num_hidden_layers,
                                  input_dim=self.first_indices.shape[0]+conditional_inp_dim,
                                  hidden_dim=hidden_dim,
                                  output_dim=self.second_indices.shape[0]*transform_function.num_params)

        # self.net = FeedForwardNet(num_hidden_layers=num_hidden_layers,
        #                           input_dim=self.first_indices.shape[0],
        #                           hidden_dim=hidden_dim,
        #                           output_dim=self.second_indices.shape[0]*transform_function.num_params)

        self.train_function = transform_function.reverse
        self.generate_function = transform_function.standard


    def forward(self, input):

        z, logdet, cond_input = input

        net_input = torch.cat((z[..., self.first_indices], cond_input), -1)
        # net_input = z[..., self.first_indices]
        net_output = self.net(net_input)

        # epsilon = z.clone().detach().requires_grad_(True)
        epsilon = torch.tensor(z)
        epsilon[..., self.second_indices], delta_logdet = \
            self.train_function( z[..., self.second_indices], net_output.view(*net_output.shape[:-1], self.second_indices.shape[0], -1) )

        return epsilon, logdet+delta_logdet, cond_input

class SCF(nn.Module) :
    def __init__(self, num_scf_layers, num_hidden_layers, hidden_dim, data_dim, conditional_inp_dim):
        super(SCF, self).__init__()
        layers = []
        reverse = False
        for i in range(num_scf_layers) :
            layers.append(SCFLayer(num_hidden_layers, hidden_dim, data_dim, conditional_inp_dim, reverse))
            reverse = not reverse
        self.scf = torch.nn.Sequential(*layers)

    def forward(self, input) :
        z, cond_input = input
        logdet = torch.zeros(z.shape[:-1], device=z.device)
        epsilon, logdet, _ = self.scf((z, logdet, cond_input))
        return epsilon, logdet

class Flow(nn.Module) :
    def __init__(self, num_scf_layers, num_hidden_layers, hidden_dim, data_dim, conditional_inp_dim):
        super(Flow, self).__init__()
        self.flow = SCF(num_scf_layers, num_hidden_layers, hidden_dim, data_dim, conditional_inp_dim)
        self.rnn = torch.nn.GRU(input_size=data_dim, hidden_size=conditional_inp_dim, batch_first=True)

    def forward(self, input):
        z = input
        rnn_output, rnn_h_n = self.rnn(z)

        # print(torch.zeros(rnn_output.shape[-1]).shape)
        # print(rnn_output[:,1:].shape)
        cond_input = torch.cat((torch.zeros(rnn_output.shape[0], 1, rnn_output.shape[2]).cuda(), rnn_output[:,1:]), dim=1)

        epsilon, logdet = self.flow((z, cond_input))
        return epsilon, logdet

class DiscreteFlow(nn.Module) :
    def __init__(self, flow_num_scf_layers, flow_num_hidden_layers, flow_hidden_dim, flow_data_dim, flow_conditional_inp_dim,
                    vae_num_max_prev_node, vae_encoder_layers=4, vae_decoder_layers=4) :
        super(DiscreteFlow, self).__init__()
        self.vae = VAE(vae_num_max_prev_node, vae_encoder_layers, vae_decoder_layers)
        self.flow = Flow(flow_num_scf_layers, flow_num_hidden_layers, flow_hidden_dim, flow_data_dim, flow_conditional_inp_dim)

    def forward(self, input, phase=None):
        data = input

        if phase == 'vae' :
            recon_batch, z, mu, logvar = self.vae(data)
            return recon_batch, mu, logvar

        elif phase == 'flow' :
            with torch.no_grad() :
                z, _, _ = self.vae.inference(data)
            epsilon, logdet = self.flow(z)
            return epsilon, logdet

        else :
            recon_batch, z, mu, logvar = self.vae(data)
            epsilon, logdet = self.flow(z)
            return recon_batch, z, mu, logvar, epsilon, logdet


