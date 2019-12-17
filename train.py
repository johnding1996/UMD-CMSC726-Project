import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import random
import os
from utils import *
from dataloader import Graph_sequence_sampler_pytorch
from vae_model import VAE

# Public args
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--N', type=int, default=2000)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--no-cuda', default=False)
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--run-name', type=str, default='test')
parser.add_argument('--output-dir', type=str, default='output')
parser.add_argument('--graph-type', type=str, default='NWS')

# Parsing
args = parser.parse_args()
setattr(args, 'save_dir', args.output_dir + '/' + args.run_name + '/saves/')
setattr(args, 'log_dir', args.output_dir + '/' + args.run_name + '/logs/')
setattr(args, 'graph_dir', args.output_dir + '/' + args.run_name + '/graphs/')
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.grpah_dir, exist_ok=True)
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# Generate graphs
Gs = create_graphs(args)
save_graphs(Gs, args.graph_dir + 'graphs.dat')
Gs_train = graphs[:int(0.8*args.N)]
Gs_test = graphs[int(0.8*args.N):]


dataset_train = GraphSampler(Gs_train, max_prev_node=args.max_prev_node, max_num_node=args.max_num_node)


dataset_val = GraphSampler(graphs_validate,max_prev_node=args.max_prev_node,max_num_node=args.max_num_node)
sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset_train) for i in range(len(dataset_train))],
                                                                num_samples=args.batch_size*args.batch_ratio, replacement=True)

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
train_iter = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=sample_strategy, **kwargs)
val_iter = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, sampler=sample_strategy, **kwargs)


model = VAE(args.n, args.max_prev_node).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, x.shape[1]*x.shape[2]), reduction='sum')
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
        recon_batch = model.decode(z)
        loss = loss_function(recon_batch, data, mu, logvar)


        #################################################################################
        # log_p_z = model.prior.evaluate(z, lengths_s=None, cond_inp_s=None)
        # loss += -log_p_z
        #################################################################################



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


# Command line entry point
if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)