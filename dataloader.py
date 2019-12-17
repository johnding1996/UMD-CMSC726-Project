import torch
import networkx as nx
import numpy as np
from utils import *


class GraphGenerator:
    def __init__(self, graph_type, N, **kwargs):
        self.N = N
        self.graph_type = graph_type
        self.Gs = self.generate(N, **kwargs)
    
    def generate(N, **kwargs):
        print("Generating Graphs, type {}".format(self.graph_type))
        if self.graph_type = 'NWS':
            return [nx.newman_watts_strogatz_graph(n=kwargs.n, k=kwargs.k, p=kwargs.p) for _ in range(N)]
        else
            raise ValueError

    def save(self, path):
        f = open(path, 'wb')
        cPickle.dump(self.__dict__, f, 2)
        f.close()
    
    def load(self, path):
        f = open(path, 'rb')
        tmp_dict = cPickle.load(f)
        f.close()
        
    def max_n(self):
        return max([self.Gs.number_of_nodes() for G in self.Gs])
    
    def As(self):
        return [np.asarray(nx.to_numpy_matrix(G)) for G in self.Gs]
        
        
    

class GraphSampler(torch.utils.data.Dataset):
    def __init__(self, gg):
        self.N = gg.N()
        self.As = gg.As()
        self.max_n = gg.max_n()
        print('Calculating m, total iteration {}'.format(iteration))
        self.max_prev_node = max(self.calc_max_prev_node(iter=iteration))
        print('m = {}'.format(self.max_prev_node))
    def __len__(self):
        return len(self.adj_all)
    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        x_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        x_batch[0,:] = 1 # the first input token is all ones
        y_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        # generate input x, y pairs
        len_batch = adj_copy.shape[0]
        x_idx = np.random.permutation(adj_copy.shape[0])
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_copy_matrix = np.asmatrix(adj_copy)
        G = nx.from_numpy_matrix(adj_copy_matrix)
        # then do bfs in the permuted G
        start_idx = np.random.randint(adj_copy.shape[0])
        x_idx = np.array(bfs_seq(G, start_idx))  
        ## convert G's adj matrix into a BFS-ordered graph's adj matrix
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]  
        ## truncate the matrix into n*M(max_prev_node)
        adj_encoded = encode_adj(adj_copy.copy(), max_prev_node=self.max_prev_node)
        # get x and y and adj
        # for small graph the rest are zero padded
        y_batch[0:adj_encoded.shape[0], :] = adj_encoded
        x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded
        return {'x':x_batch,'y':y_batch, 'len':len_batch}

    def calc_max_prev_node(self, iter=20000, topk=10):
        max_prev_node = []
        for i in range(iter):
            if i % (iter / 5) == 0:
                print('iter {} times'.format(i))
                
            # randomly pick an A
            adj_idx = np.random.randint(self.N)
            adj_copy = self.As[adj_idx].copy()
            n = adj_copy.shape[0]
            x_idx = np.random.permutation(n)
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            # encode adj
            adj_encoded = encode_adj_flexible(adj_copy.copy())
            max_encoded_len = max([len(adj_encoded[i]) for i in range(len(adj_encoded))])
            max_prev_node.append(max_encoded_len)
        max_prev_node = sorted(max_prev_node)[-1*topk:]
        return max_prev_node
