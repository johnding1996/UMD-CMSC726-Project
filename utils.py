import pickle
import networkx as nx
import numpy as np

def create_graphs(args):
    graphs=[]
    # synthetic graphs
    if args.graph_type=='ladder':
        graphs = []
        for i in range(100, 201):
            graphs.append(nx.ladder_graph(i))
        args.max_prev_node = 10
    elif args.graph_type=='ladder_small':
        graphs = []
        for i in range(2, 11):
            graphs.append(nx.ladder_graph(i))
        args.max_prev_node = 10
    elif args.graph_type=='tree':
        graphs = []
        for i in range(2,5):
            for j in range(3,5):
                graphs.append(nx.balanced_tree(i,j))
        args.max_prev_node = 256
    # elif args.graph_type=='caveman':
    #     # graphs = []
    #     # for i in range(5,10):
    #     #     for j in range(5,25):
    #     #         for k in range(5):
    #     #             graphs.append(nx.relaxed_caveman_graph(i, j, p=0.1))
    #     graphs = []
    #     for i in range(2, 3):
    #         for j in range(30, 81):
    #             for k in range(10):
    #                 graphs.append(caveman_special(i,j, p_edge=0.3))
    #     args.max_prev_node = 100
    # elif args.graph_type=='caveman_small':
    #     # graphs = []
    #     # for i in range(2,5):
    #     #     for j in range(2,6):
    #     #         for k in range(10):
    #     #             graphs.append(nx.relaxed_caveman_graph(i, j, p=0.1))
    #     graphs = []
    #     for i in range(2, 3):
    #         for j in range(6, 11):
    #             for k in range(20):
    #                 graphs.append(caveman_special(i, j, p_edge=0.8)) # default 0.8
    #     args.max_prev_node = 20
    # elif args.graph_type=='caveman_small_single':
    #     # graphs = []
    #     # for i in range(2,5):
    #     #     for j in range(2,6):
    #     #         for k in range(10):
    #     #             graphs.append(nx.relaxed_caveman_graph(i, j, p=0.1))
    #     graphs = []
    #     for i in range(2, 3):
    #         for j in range(8, 9):
    #             for k in range(100):
    #                 graphs.append(caveman_special(i, j, p_edge=0.5))
    #     args.max_prev_node = 20
    # elif args.graph_type.startswith('community'):
    #     num_communities = int(args.graph_type[-1])
    #     print('Creating dataset with ', num_communities, ' communities')
    #     c_sizes = np.random.choice([12, 13, 14, 15, 16, 17], num_communities)
    #     #c_sizes = [15] * num_communities
    #     for k in range(3000):
    #         graphs.append(n_community(c_sizes, p_inter=0.01))
    #     args.max_prev_node = 80
    elif args.graph_type=='grid':
        graphs = []
        for i in range(10,20):
            for j in range(10,20):
                graphs.append(nx.grid_2d_graph(i,j))
        args.max_prev_node = 40
    elif args.graph_type=='grid_small':
        graphs = []
        for i in range(2,5):
            for j in range(2,6):
                graphs.append(nx.grid_2d_graph(i,j))
        args.max_prev_node = 15
    elif args.graph_type=='barabasi':
        graphs = []
        for i in range(20,30):
             for j in range(4,5):
                graphs.append(nx.barabasi_albert_graph(i,j))
        args.max_prev_node = 30
    elif args.graph_type=='barabasi_small':
        graphs = []
        for i in range(4,21):
             for j in range(3,4):
                 for k in range(10):
                    graphs.append(nx.barabasi_albert_graph(i,j))
        args.max_prev_node = 20
    elif args.graph_type=='grid_big':
        graphs = []
        for i in range(36, 46):
            for j in range(36, 46):
                graphs.append(nx.grid_2d_graph(i, j))
        args.max_prev_node = 90

    # elif 'barabasi_noise' in args.graph_type:
    #     graphs = []
    #     for i in range(100,101):
    #         for j in range(4,5):
    #             for k in range(500):
    #                 graphs.append(nx.barabasi_albert_graph(i,j))
    #     graphs = perturb_new(graphs,p=args.noise/10.0)
    #     args.max_prev_node = 99

    return graphs

# Model utility functions
# ------------------------------------------------------------------------------------------------------------------------------
# save a list of graphs
def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)


def bfs_seq(G, start_id):
    '''
    get a bfs node sequence
    :param G:
    :param start_id:
    :return:
    '''
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                #### a wrong example, should not permute here!
                # shuffle(neighbor)
                next = next + neighbor
        output = output + next
        start = next
    return output



def encode_adj(adj, max_prev_node=10, is_full = False):
    '''

    :param adj: n*n, rows means time step, while columns are input dimension
    :param max_degree: we want to keep row number, but truncate column numbers
    :return: n*M(max_prev_node)
    '''
    if is_full:
        max_prev_node = adj.shape[0]-1

    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]

    # use max_prev_node to truncate
    # note: now adj is a (n-1)*(n-1) matrix
    adj_output = np.zeros((adj.shape[0], max_prev_node))
    for i in range(adj.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + input_start - input_end
        output_end = max_prev_node
        adj_output[i, output_start:output_end] = adj[i, input_start:input_end]
        adj_output[i,:] = adj_output[i,:][::-1] # reverse order

    return adj_output

def decode_adj(adj_output):
    '''
        recover to adj from adj_output
        note: here adj_output have shape (n-1)*m
    '''
    max_prev_node = adj_output.shape[1]
    adj = np.zeros((adj_output.shape[0], adj_output.shape[0]))
    for i in range(adj_output.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
        output_end = max_prev_node
        adj[i, input_start:input_end] = adj_output[i,::-1][output_start:output_end] # reverse order
    adj_full = np.zeros((adj_output.shape[0]+1, adj_output.shape[0]+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full


def encode_adj_flexible(adj):
    '''
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return: not a matrix, but a list of adj vectors.
    '''
    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]

    adj_output = []
    input_start = 0
    for i in range(adj.shape[0]):
        input_end = i + 1
        adj_slice = adj[i, input_start:input_end]
        adj_output.append(adj_slice)
        non_zero = np.nonzero(adj_slice)[0]
        input_start = input_end-len(adj_slice)+np.amin(non_zero)

    return adj_output


def decode_adj_flexible(adj_output):
    '''
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return:
    '''
    adj = np.zeros((len(adj_output), len(adj_output)))
    for i in range(len(adj_output)):
        output_start = i+1-len(adj_output[i])
        output_end = i+1
        adj[i, output_start:output_end] = adj_output[i]
    adj_full = np.zeros((len(adj_output)+1, len(adj_output)+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full

