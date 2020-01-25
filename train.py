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
from utils import *
from data import GraphDataset
from models import DiscreteFlow
import math


parser = argparse.ArgumentParser(description='Graph Discrete Flow')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--batch_ratio', type=int,
                    default=80)  # how many batches of samples per epoch, default 32, e.g., 1 epoch = 32 batches

parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true',
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=16, metavar='N',
                    help='how many batches to wait before logging training status')
# parser.add_argument('--output-dir', type=str, default='output')
parser.add_argument('--output-dir', type=str, default='/cmlscratch/kong/records/flow')
parser.add_argument('--run-name', type=str, default='grid0')
parser.add_argument('--graph-type', type=str, default='grid') #barabasi_test


args = parser.parse_args()
setattr(args, 'savedir', args.output_dir + '/' + args.run_name + '/saves/')
setattr(args, 'logdir', args.output_dir + '/' + args.run_name + '/logs/')
setattr(args, 'fname_train', args.graph_type + '_train_')
setattr(args, 'fname_test', args.graph_type + '_test_')
setattr(args, 'graph_save_path', args.output_dir + '/graphs')
args.cuda = not args.no_cuda and torch.cuda.is_available()


os.makedirs(args.savedir, exist_ok=True)
os.makedirs(args.logdir, exist_ok=True)
torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")


graphs = create_graphs(args)
random.shuffle(graphs)
graphs_len = len(graphs)
graphs_test = graphs[int(0.8 * graphs_len):]
graphs_train = graphs[0:int(0.8*graphs_len)]
# graphs_validate = graphs[0:int(0.2*graphs_len)]
save_graph_list(graphs, args.graph_save_path + args.fname_train + '0.dat')
save_graph_list(graphs, args.graph_save_path + args.fname_test + '0.dat')
print('train and test graphs saved at: ', args.graph_save_path + args.fname_test + '0.dat')


dataset_train = GraphDataset(graphs_train,max_prev_node=args.max_prev_node)
# dataset_val = GraphDataset(graphs_validate,max_prev_node=args.max_prev_node)
dataset_test = GraphDataset(graphs_test,max_prev_node=args.max_prev_node)

# print(dataset_train.calc_max_prev_node(iter=1000))
# exit()

sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset_train) for i in range(len(dataset_train))],
                                                                num_samples=len(dataset_train), replacement=True)

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
train_iter = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=sample_strategy, **kwargs)
test_iter = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, sampler=sample_strategy, **kwargs)

model = DiscreteFlow(flow_num_scf_layers=8, flow_num_hidden_layers=4, flow_hidden_dim=32,
                     flow_data_dim=args.max_prev_node, flow_conditional_inp_dim=args.max_prev_node//4, #args.max_prev_node//4
                     vae_num_max_prev_node=args.max_prev_node, batch_size=args.batch_size).to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-3)
BCE_loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean')

# Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss_function(recon_x, x, mu, logvar):
    BCE = BCE_loss_function(recon_x.reshape(recon_x.shape[0], -1), x.reshape(x.shape[0], -1))
    mu  = mu.reshape(mu.shape[0], -1)
    logvar = logvar.reshape(logvar.shape[0], -1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def flow_loss_function(epsilon, logdet) :
    # print(epsilon.shape)
    # print(logdet.shape)
    # exit()
    log_p_eps = -1/2*(math.log(2*math.pi) + epsilon.pow(2)).sum()
    log_p_eps *= -1

    log_det = logdet.sum()
    log_det *= -1

    log_p_z = log_p_eps + log_det
    ratio = 1/logdet.shape[1]
    # ratio = 1

    # print("-log_p_eps: {}".format(log_p_eps.sum()))
    # print("logdet: {}".format(logdet.sum()))
    # return -log_p_z.mean()
    # return -log_p_z.mean()/logdet.shape[1]
    return ratio*log_p_z, ratio*log_p_eps, ratio*log_det

def train(epoch):
    model.train()
    train_loss, train_log_det, train_log_p = 0, 0, 0

    for batch_idx, data in enumerate(train_iter):
        data = data.float().to(device)
        optimizer.zero_grad()

        # print(vae_loss.shape)
        # print(flow_loss.shape)
        # exit()

        two_phase = False
        if epoch <= args.epochs//2: #args.epochs//2
            recon_batch, mu, logvar = model(data, phase='vae')
            loss = vae_loss_function(recon_batch, data, mu, logvar)
        else :
            if two_phase : #only train flow model
                epsilon, logdet = model(data, phase='flow')
                loss, log_p, log_det = flow_loss_function(epsilon, logdet)
                train_log_det += log_det.item()
                train_log_p += log_p.item()
            else : #train the whole model
                # for g in optimizer.param_groups:
                #     g['lr'] = 1e-7
                recon_batch, z, mu, logvar, epsilon, logdet = model(data)
                loss = vae_loss_function(recon_batch, data, mu, logvar)
                loss_flow, log_p, log_det = flow_loss_function(epsilon, logdet)
                loss += loss_flow
                train_log_det += log_det.item()
                train_log_p += log_p.item()


        loss.backward()
        train_loss += loss.item()


        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_iter.dataset),
                100. * batch_idx / len(train_iter),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_iter.dataset)))
    print('Average logdet: {:.4f} Average log_p: {:.4f}'.format(
        train_log_det/len(train_iter.dataset), train_log_p/len(train_iter.dataset)))

# def train(epoch):
#     model.train()
#     train_loss = 0
#
#     for batch_idx, data in enumerate(train_iter):
#         data = data.float().to(device)
#
#         if batch_idx%10 == 0 :
#             optimizer.zero_grad()
#
#         recon_batch, z, mu, logvar, epsilon, logdet = model(data)
#         loss = vae_loss_function(recon_batch, data, mu, logvar)
#         if epoch > args.epochs//2:
#             loss += flow_loss_function(epsilon, logdet)
#         train_loss += loss.item()
#
#         if batch_idx % 10 == 9:
#             loss.backward()
#             optimizer.step()
#
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_iter.dataset),
#                 100. * batch_idx / len(train_iter),
#                 loss.item() / len(data)))
#
#     print('====> Epoch: {} Average loss: {:.4f}'.format(
#           epoch, train_loss / len(train_iter.dataset)))

def test():
    model.eval()
    num_of_nodes = [40,50,60]
    with torch.no_grad():
        graph_bfs_reps = model.generate(num_of_nodes, device)
        print(graph_bfs_reps[0].shape)
    # test_loss = 0
    # with torch.no_grad():
    #     for i, data in enumerate(test_iter):
    #         data = data.to(device)
    #         recon_batch, mu, logvar = model(data)
    #         test_loss += loss_function(recon_batch, data, mu, logvar).item()
    #         if i == 0:
    #             n = min(data.size(0), 8)
    #             comparison = torch.cat([data[:n],
    #                                   recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
    #             save_image(comparison.cpu(),
    #                      'results/reconstruction_' + str(epoch) + '.png', nrow=n)
    #
    # test_loss /= len(test_loader.dataset)
    # print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)

    test()
        # test(epoch)
        # with torch.no_grad():
        #     sample = torch.randn(64, 20).to(device)
        #     sample = model.decode(sample).cpu()
        #     save_image(sample.view(64, 1, 28, 28),
        #                'results/sample_' + str(epoch) + '.png')