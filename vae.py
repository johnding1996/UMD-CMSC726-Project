from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import random
import os
from utils import save_graph_list
from data import create_graphs
from data.dataloader import Graph_sequence_sampler_pytorch
from lstm_flow import AFPrior


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--batch_ratio', type=int,
                    default=1)  # how many batches of samples per epoch, default 32, e.g., 1 epoch = 32 batches

parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--output_dir', type=str, default='output')
parser.add_argument('--run_name', type=str, default='vae_test')
parser.add_argument('--graph_type', type=str, default='grid')
parser.add_argument('--max_prev_node', type=int)

args = parser.parse_args()
setattr(args, 'savedir', args.output_dir + '/' + args.run_name + '/saves/')
setattr(args, 'logdir', args.output_dir + '/' + args.run_name + '/logs/')
setattr(args, 'fname_train', args.graph_type + '_train_')
setattr(args, 'fname_test', args.graph_type + '_test_')
setattr(args, 'graph_save_path', args.output_dir + '/graphs')

os.makedirs(args.savedir, exist_ok=True)
os.makedirs(args.logdir, exist_ok=True)
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

# kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)


graphs = create_graphs.create_graphs(args)
random.shuffle(graphs)
graphs_len = len(graphs)
graphs_test = graphs[int(0.8 * graphs_len):]
graphs_train = graphs[0:int(0.8*graphs_len)]
graphs_validate = graphs[0:int(0.2*graphs_len)]
args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])


save_graph_list(graphs, args.graph_save_path + args.fname_train + '0.dat')
save_graph_list(graphs, args.graph_save_path + args.fname_test + '0.dat')
print('train and test graphs saved at: ', args.graph_save_path + args.fname_test + '0.dat')


dataset_train = Graph_sequence_sampler_pytorch(graphs_train,max_prev_node=args.max_prev_node,max_num_node=args.max_num_node)
dataset_val = Graph_sequence_sampler_pytorch(graphs_validate,max_prev_node=args.max_prev_node,max_num_node=args.max_num_node)
sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset_train) for i in range(len(dataset_train))],
                                                                num_samples=args.batch_size*args.batch_ratio, replacement=True)

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
train_iter = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=sample_strategy, **kwargs)
val_iter = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, sampler=sample_strategy, **kwargs)


hiddenflow_params = {'nohiddenflow': True,
                     'hiddenflow_layers': 2,
                     'hiddenflow_units': 100,
                     'hiddenflow_flow_layers': 5,
                     'hiddenflow_scf_layers': True}

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

        ###########################################################################################
        self.prior = AFPrior(hidden_size=self.M, zsize=self.N, dropout_p=0, dropout_locations=['prior_rnn'], prior_type='AF', num_flow_layers=1, rnn_layers=2,
                              transform_function='nlsq', hiddenflow_params=hiddenflow_params)
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


model = VAE(args.max_num_node, args.max_prev_node).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, x.shape[1]*x.shape[2]), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_iter):
        data = data['x'].float()
        data = data.to(device)
        optimizer.zero_grad()
        # recon_batch, mu, logvar = model(data)
        z, mu, logvar = model.inference(data)
        recon_batch, _, _ = model.decode(z)
        loss = loss_function(recon_batch, data, mu, logvar)


        #################################################################################
        log_p_z = model.prior.evaluate(z, lengths_s=None, cond_inp_s=None)
        #################################################################################

        loss += -log_p_z

        loss.backward()
        train_loss += loss.item()
        print(loss.item())
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_iter.dataset),
                100. * batch_idx / len(train_iter),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_iter.dataset)))


# def test(epoch):
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for i, (data, _) in enumerate(test_loader):
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()
#             if i == 0:
#                 n = min(data.size(0), 8)
#                 comparison = torch.cat([data[:n],
#                                       recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
#                 save_image(comparison.cpu(),
#                          'results/reconstruction_' + str(epoch) + '.png', nrow=n)
#
#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        # test(epoch)
        # with torch.no_grad():
        #     sample = torch.randn(64, 20).to(device)
        #     sample = model.decode(sample).cpu()
        #     save_image(sample.view(64, 1, 28, 28),
        #                'results/sample_' + str(epoch) + '.png')