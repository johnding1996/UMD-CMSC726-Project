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
import eval.stats
import numpy as np


parser = argparse.ArgumentParser(description='Graph Discrete Flow')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--batch_ratio', type=int,
                    default=80)  # how many batches of samples per epoch, default 32, e.g., 1 epoch = 32 batches

parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true',
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=16, metavar='N',
                    help='how many batches to wait before logging training status')
# parser.add_argument('--output-dir', type=str, default='output')
parser.add_argument('--output-dir', type=str, default='/cmlscratch/kong/records/flow')
parser.add_argument('--run-name', type=str, default='run6')
parser.add_argument('--graph-type', type=str, default='grid') #barabasi_test


parser.add_argument('--flow-num-scf-layers', type=int, default=16)
parser.add_argument('--flow-num-hidden-layers', type=int, default=4)
parser.add_argument('--flow-hidden-dim', type=int, default=32)
parser.add_argument('--lr', type=float, default=5e-3)


args = parser.parse_args()
setattr(args, 'savedir', args.output_dir + '/' + args.run_name + '/saves')
setattr(args, 'logdir', args.output_dir + '/' + args.run_name + '/logs')
setattr(args, 'fname_train', args.graph_type + '_train_')
setattr(args, 'fname_test', args.graph_type + '_test_')

setattr(args, 'fname_pred', args.graph_type + '_pred_')

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

model = DiscreteFlow(flow_num_scf_layers=args.flow_num_scf_layers, flow_num_hidden_layers=args.flow_num_hidden_layers,
                     flow_hidden_dim=args.flow_hidden_dim, flow_data_dim=args.max_prev_node//2, flow_conditional_inp_dim=None, #args.max_prev_node//4
                     vae_num_max_prev_node=args.max_prev_node, batch_size=args.batch_size).to(device)
model_name = 'DiscreteFlow-{}-SCFLayers{}-FlowHiddenLayers{}-FlowHiddenDim{}-Epochs{}-LR{}-{}.pt'.format(
    args.graph_type, args.flow_num_scf_layers, args.flow_num_hidden_layers, args.flow_hidden_dim, args.epochs, args.lr, args.run_name)
log_name = 'DiscreteFlow-{}-SCFLayers{}-FlowHiddenLayers{}-FlowHiddenDim{}-Epochs{}-LR{}-{}.txt'.format(
    args.graph_type, args.flow_num_scf_layers, args.flow_num_hidden_layers, args.flow_hidden_dim, args.epochs, args.lr, args.run_name)
optimizer = optim.SGD(model.parameters(), lr=args.lr)
BCE_loss_function = torch.nn.BCELoss(reduction='mean')
MSE_loss_function = torch.nn.MSELoss(reduction='mean')

# Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss_function(recon_x, x, mu, logvar):
    BCE = BCE_loss_function(recon_x.reshape(recon_x.shape[0], -1), x.reshape(x.shape[0], -1))
    # MSE = MSE_loss_function(recon_x.reshape(recon_x.shape[0], -1), x.reshape(x.shape[0], -1))
    # mu  = mu.reshape(mu.shape[0], -1)
    # logvar = logvar.reshape(logvar.shape[0], -1)
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # return BCE + (1e-3)*KLD
    return BCE

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

def train(epoch, f):
    model.train()
    train_loss, train_log_det, train_log_p = 0, 0, 0

    for batch_idx, data in enumerate(train_iter):
        data = data.float().to(device)
        optimizer.zero_grad()

        # print(vae_loss.shape)
        # print(flow_loss.shape)
        # exit()

        two_phase = False
        if epoch <= 200: #args.epochs//2
            recon_batch, mu, logvar = model(data, phase='vae')
            loss = vae_loss_function(recon_batch, data, mu, logvar)
        else :
            if two_phase : #only train flow model
                epsilon, logdet = model(data, phase='flow')
                loss, log_p, log_det = flow_loss_function(epsilon, logdet)
                train_log_det += log_det.item()
                train_log_p += log_p.item()
            else : #train the whole model
                for g in optimizer.param_groups:
                    g['lr'] = 1e-3
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


    tmp = '====> Epoch: {} Average loss: {:.4f}\n'.format(epoch, train_loss / len(train_iter.dataset))
    tmp += '      Average logdet: {:.4f} Average log_p: {:.4f}\n'.format(
        train_log_det/len(train_iter.dataset), train_log_p/len(train_iter.dataset))
    f.write(tmp)
    print(tmp)

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

def test(f):
    model.eval()

    # num_of_nodes = np.ones(1000, dtype=np.int)*180

    # for g in graphs_train:
    #     print(g.number_of_nodes())
    #
    # exit()

    num_of_nodes = []
    for graph_test in graphs_test :
        num_of_nodes.append(graph_test.number_of_nodes()-2)

    with torch.no_grad():
        graphs_bfs_reps = model.generate(num_of_nodes, device)

        # count = 0
        # for graph_bfs_rep in graphs_bfs_reps :
        #     if True in (graph_bfs_rep > 0) :
        #         count += 1
        # print(count/1000)
        # exit()

        graphs_pre = convert_graph(graphs_bfs_reps)

        # for g in graphs_bfs_reps :
        #     print(g.shape)
        #     exit()

        # mmd_degree = eval.stats.degree_stats(graphs_test, graphs_pre)
        # print(mmd_degree)

        tmp = ''
        try:
            mmd_degree = eval.stats.degree_stats(graphs_test, graphs_pre)
            tmp += 'mmd degree:{}\n'.format(mmd_degree)
        except:
            print("degree exploded")

        try:
            mmd_clustering = eval.stats.clustering_stats(graphs_test, graphs_pre)
            tmp += 'mmd clustering:{}\n'.format(mmd_clustering)
        except:
            print('clustering exploded')

        try:
            mmd_4orbits = eval.stats.orbit_stats_all(graphs_test, graphs_pre)
            tmp += 'mmd 4orbits:{}\n'.format(mmd_4orbits)
        except:
            print('orbits exploded')

        f.write(tmp)
        print(tmp)
    # pred_graphs = convert_graph(recon_batch)
    # # print(pred_graphs[0].nodes())
    # # print(pred_graphs[0].edges())
    # save_graph_list(pred_graphs, args.graph_save_path + args.fname_pred + str(epoch) + '.dat')



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

    with open(os.path.join(args.logdir, log_name), 'w') as f:
        for epoch in range(1, args.epochs + 1):
            train(epoch, f)

            if epoch > 200 and epoch % 500 == 0 :
                torch.save(model.state_dict(), os.path.join(args.savedir, str(epoch)+model_name))
                test(f)

        # model.load_state_dict(torch.load(os.path.join(args.savedir, model_name)))
        # test(epoch)
        # with torch.no_grad():
        #     sample = torch.randn(64, 20).to(device)
        #     sample = model.decode(sample).cpu()
        #     save_image(sample.view(64, 1, 28, 28),
        #                'results/sample_' + str(epoch) + '.png')