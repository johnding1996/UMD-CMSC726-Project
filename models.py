import torch
from torch import nn
from torch.nn import functional as F
from transformations import NLSq, Affine

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

class VAE(nn.Module):
    def __init__(self, num_max_prev_node, encoder_layers, decoder_layers):
        super(VAE, self).__init__()

        self.input_size = num_max_prev_node #M
        self.rnn1 = torch.nn.GRU(input_size=self.input_size, hidden_size=self.input_size//2, num_layers=encoder_layers, batch_first=True)
        self.rnn21 = torch.nn.GRU(input_size=self.input_size, hidden_size=self.input_size, num_layers=1, batch_first=True)
        self.rnn22 = torch.nn.GRU(input_size=self.input_size, hidden_size=self.input_size, num_layers=1, batch_first=True)
        self.rnn3 = torch.nn.GRU(input_size=self.input_size, hidden_size=self.input_size, num_layers=decoder_layers, batch_first=True)

        self.fout = FeedForwardNet(4, self.input_size//2, self.input_size, self.input_size)

        #flow model
        ###########################################################################################
        # self.prior = AFPrior(hidden_size=self.M, zsize=self.N, dropout_p=0, dropout_locations=['prior_rnn'], prior_type='AF', num_flow_layers=1, rnn_layers=2,
        #                       transform_function='nlsq', hiddenflow_params=hiddenflow_params)
        ###########################################################################################

    def encode(self, x): #  (batch, seq_len, input_size)
        output, h_n = self.rnn1(x)
        # mu, _ = self.rnn21(output) # (output, h_n)?    (batch, seq_len, input_size)
        # logvar, _ = self.rnn22(output) # (output, h_n)?  (batch, seq_len, input_size)
        # return mu, logvar
        return output[:,:-1]


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar.reshape(logvar.shape[0], -1) )
        eps = torch.randn_like(std)
        return mu.reshape(mu.shape[0], -1) + eps*std

    def decode(self, z):
        # output, _ = self.rnn3(z)
        tmp = torch.zeros((1,1,self.input_size), device=z.device)
        tmp[0,0,0] = 1

        output = self.fout(z)
        output = torch.sigmoid(output)
        return torch.cat((tmp, output), dim=1)


    def inference(self, x):
        # mu, logvar = self.encode(x)
        mu = self.encode(x)
        # z = self.reparameterize(mu, logvar)
        # return z.view(-1, mu.shape[1], mu.shape[2]), mu, logvar
        return mu

    def forward(self, x):
        # z, mu, logvar = self.inference(x)
        # return self.decode(z), z, mu, logvar
        z = self.inference(x)
        return self.decode(z), z, None, None


class SCFLayer(nn.Module) :
    def __init__(self, num_hidden_layers, hidden_dim, data_dim, conditional_inp_dim, reverse, transform_function=Affine):
        super(SCFLayer, self).__init__()
        self.data_dim = data_dim
        self.conditional_inp_dim = conditional_inp_dim

        indices = torch.arange(self.data_dim)
        if reverse :
            self.first_indices = indices[self.data_dim//2:]
            self.second_indices = indices[:self.data_dim//2]
        else :
            self.first_indices = indices[:self.data_dim//2]
            self.second_indices = indices[self.data_dim//2:]

        if conditional_inp_dim :
            self.net = FeedForwardNet(num_hidden_layers=num_hidden_layers,
                                      input_dim=self.first_indices.shape[0]+conditional_inp_dim,
                                      hidden_dim=hidden_dim,
                                      output_dim=self.second_indices.shape[0]*transform_function.num_params)
            self.rnn = torch.nn.GRU(input_size=data_dim, hidden_size=conditional_inp_dim, batch_first=True)
        else:
            self.net = FeedForwardNet(num_hidden_layers=num_hidden_layers,
                                      input_dim=self.first_indices.shape[0],
                                      hidden_dim=hidden_dim,
                                      output_dim=self.second_indices.shape[0]*transform_function.num_params)

        self.train_function = transform_function.standard
        self.generate_function = transform_function.reverse


    def forward(self, input):

        z, logdet = input
        if self.conditional_inp_dim :
            rnn_output, rnn_h_n = self.rnn(z)
            # cond_input = torch.cat((torch.zeros(rnn_output.shape[0], 1, rnn_output.shape[2]).cuda(),
            #                         rnn_output[:, 1:]), dim=1)
            cond_input = torch.cat((torch.zeros((rnn_output.shape[0], 1, rnn_output.shape[2]), device=z.device),
                                    rnn_output[:, 1:]), dim=1)
            net_input = torch.cat((z[..., self.first_indices], cond_input), -1)
            net_output = self.net(net_input)

            # epsilon = z.detach().requires_grad_(True)
            # epsilon = torch.tensor(z)

            output = torch.zeros(z.shape, device=z.device)
            output[..., self.first_indices] = z[..., self.first_indices]
            output[..., self.second_indices], delta_logdet = \
                self.train_function( z[..., self.second_indices], net_output.view(*net_output.shape[:-1], self.second_indices.shape[0], -1) )
        else :
            net_input = z[..., self.first_indices]
            net_output = self.net(net_input)
            # epsilon = torch.tensor(z)
            # epsilon = z.clone().detach().requires_grad_(True)

            output = torch.zeros(z.shape, device=z.device)
            output[..., self.first_indices] = z[..., self.first_indices]
            output[..., self.second_indices], delta_logdet = \
                self.train_function( z[..., self.second_indices], net_output.view(*net_output.shape[:-1], self.second_indices.shape[0], -1) )

        return output, logdet+delta_logdet

    def generate(self, input):
        epsilon = input
        if self.conditional_inp_dim:
            for i in range(epsilon.shape[1]) :
                if i == 0 :
                    cond_input = torch.zeros((epsilon.shape[0], 1, self.conditional_inp_dim), device=epsilon.device)

                    # print(epsilon.shape)
                    # print(epsilon[..., 0:1, self.first_indices].shape)
                    # print(cond_input.shape)
                    # exit()

                    net_input = torch.cat((epsilon[..., 0:1, self.first_indices], cond_input), -1)
                    net_output = self.net(net_input)

                    output = torch.zeros(epsilon.shape, device=epsilon.device)
                    output[..., self.first_indices] = epsilon[..., self.first_indices]
                    output[..., 0:1, self.second_indices] , _ = \
                        self.generate_function( epsilon[..., 0:1, self.second_indices], net_output.view(*net_output.shape[:-1], self.second_indices.shape[0], -1))
                    cond_input, _ = self.rnn(epsilon[..., 0:1, :])
                else:
                    net_input = torch.cat((epsilon[..., i:i+1, self.first_indices], cond_input), -1)
                    net_output = self.net(net_input)

                    output = torch.zeros(epsilon.shape, device=epsilon.device)
                    output[..., self.first_indices] = epsilon[..., self.first_indices]
                    output[..., i:i+1, self.second_indices], _ = \
                        self.generate_function(epsilon[..., i:i+1, self.second_indices], net_output.view(*net_output.shape[:-1], self.second_indices.shape[0], -1))
                    cond_input, _ = self.rnn(epsilon[..., i:i+1, :], cond_input)

        else:
            net_input = epsilon[..., self.first_indices]
            net_output = self.net(net_input)

            output = torch.zeros(epsilon.shape, device=epsilon.device)
            output[..., self.first_indices] = epsilon[..., self.first_indices]
            output[..., self.second_indices], _ = \
                self.generate_function( epsilon[..., self.second_indices], net_output.view(*net_output.shape[:-1], self.second_indices.shape[0], -1) )

        return output


class SCF(nn.Module) :
    def __init__(self, num_scf_layers, num_hidden_layers, hidden_dim, data_dim, conditional_inp_dim):
        super(SCF, self).__init__()
        self.layers = []
        reverse = False
        for i in range(num_scf_layers) :
            self.layers.append(SCFLayer(num_hidden_layers, hidden_dim, data_dim, conditional_inp_dim, reverse))
            reverse = not reverse
        self.scf = torch.nn.Sequential(*self.layers)

    def forward(self, input) :
        z = input
        logdet_init = torch.zeros(z.shape[:-1], device=z.device)
        epsilon, logdet = self.scf((z, logdet_init))
        return epsilon, logdet

    def generate(self, input) :
        epsilon = input
        for layer in reversed(self.layers) :
            epsilon = layer.generate(epsilon)
        return epsilon


# class Flow(nn.Module) :
#     def __init__(self, num_scf_layers, num_hidden_layers, hidden_dim, data_dim, conditional_inp_dim):
#         super(Flow, self).__init__()
#         self.flow = SCF(num_scf_layers, num_hidden_layers, hidden_dim, data_dim, conditional_inp_dim)
#         # self.rnn = torch.nn.GRU(input_size=data_dim, hidden_size=conditional_inp_dim, batch_first=True)
#
#     def forward(self, input):
#         z = input
#         # rnn_output, rnn_h_n = self.rnn(z)
#
#         # print(torch.zeros(rnn_output.shape[-1]).shape)
#         # print(rnn_output[:,1:].shape)
#         # cond_input = torch.cat((torch.zeros(rnn_output.shape[0], 1, rnn_output.shape[2]).cuda(), rnn_output[:,1:]), dim=1)
#
#         epsilon, logdet = self.flow((z, None))
#         return epsilon, logdet
#
#     # def generate(self, input):


class DiscreteFlow(nn.Module) :
    def __init__(self, flow_num_scf_layers, flow_num_hidden_layers, flow_hidden_dim, flow_data_dim, flow_conditional_inp_dim,
                    vae_num_max_prev_node, batch_size, vae_encoder_layers=4, vae_decoder_layers=4) :
        super(DiscreteFlow, self).__init__()
        self.vae = VAE(vae_num_max_prev_node, vae_encoder_layers, vae_decoder_layers)
        self.flow = SCF(flow_num_scf_layers, flow_num_hidden_layers, flow_hidden_dim, flow_data_dim, flow_conditional_inp_dim)
        self.flow_data_dim = flow_data_dim
        self.batch_size = batch_size

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

    def generate(self, num_of_nodes, device):
        graph_bfs_reps = list()
        for num_of_node in num_of_nodes :
            epsilon = torch.randn((self.batch_size, num_of_node, self.flow_data_dim), device=device)
            z = self.flow.generate(epsilon)

            graph_bfs_rep = self.vae.decode(z)
            graph_bfs_reps.append(graph_bfs_rep)
        return graph_bfs_reps