"""
1. identify current graph_type and related directories, load test data
2. according to step1, generate a batch of predicted data
3. compare the pred_data to test_data w.r.t MMD and other metrics
"""

import argparse
import torch
import torch.utils.data
import os
import eval.stats
from utils import *
from baselines.baseline_simple import *
from data import GraphDataset
from models import DiscreteFlow
from random import shuffle

def find_nearest_idx(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def clean_graphs(graph_real, graph_pred):
    ''' Selecting graphs generated that have the similar sizes.
    It is usually necessary for GraphRNN-S version, but not the full GraphRNN model.
    '''
    shuffle(graph_real)
    shuffle(graph_pred)

    # get length
    real_graph_len = np.array([len(graph_real[i]) for i in range(len(graph_real))])
    pred_graph_len = np.array([len(graph_pred[i]) for i in range(len(graph_pred))])

    # select pred samples
    # The number of nodes are sampled from the similar distribution as the training set
    pred_graph_new = []
    pred_graph_len_new = []
    for value in real_graph_len:
        pred_idx = find_nearest_idx(pred_graph_len, value)
        pred_graph_new.append(graph_pred[pred_idx])
        pred_graph_len_new.append(pred_graph_len[pred_idx])

    return graph_real, pred_graph_new

def evaluate(fname_output, model_name, dataset_name, args, is_clean=True, epoch_start=50, epoch_end=80, epoch_step=10):
    with open(fname_output, 'w+') as f:
        f.write('epoch,degree_validate,clustering_validate,orbits4_validate,degree_test,clustering_test,orbits4_test\n')

        # read real graph
        if model_name=='Internal' or model_name=='Noise' or model_name=='B-A' or model_name=='E-R':
            # fname_test = args.input_dir + 'GraphRNN_MLP' + '_' + dataset_name + '_' + str(args.num_layers) + '_' + '_test_' + str(0) + '.dat'
            fname_test = args.graph_save_path + dataset_name + '_test_' + '0.dat'
        elif 'Baseline' in model_name:
            fname_test = args.input_dir + model_name + '_' + dataset_name + '_' + str(64) + '_test_' + str(0) + '.dat'
        else:
            fname_test = args.graph_save_path + dataset_name + '_test_' + '0.dat'
        try:
        graph_test = load_graph_list(fname_test,is_real=False)
        except:
            print('Not found: ' + fname_test)
            return None

        graph_test_len = len(graph_test)
        graph_train = graph_test[0:int(0.8 * graph_test_len)] # train
        graph_validate = graph_test[0:int(0.2 * graph_test_len)] # validate
        graph_test = graph_test[int(0.8 * graph_test_len):] # test on a hold out test set

        graph_test_aver = 0
        for graph in graph_test:
            graph_test_aver+=graph.number_of_nodes()
        graph_test_aver /= len(graph_test)
        print('test average len',graph_test_aver)


        # get performance for proposed approaches
        if 'GraphRNN' in model_name:
            # read test graph
            for epoch in range(epoch_start,epoch_end,epoch_step):
                # get filename
                fname_pred = args.graph_save_path + dataset_name + '_pred_' + str(epoch) + '.dat'
                # load graphs
                try:
                    graph_pred = load_graph_list(fname_pred,is_real=False) # default False
                    print(fname_pred)
                    # print(graph_pred[0].nodes())
                except:
                    print('Not found: '+ fname_pred)
                    continue
                # clean graphs
                if is_clean:
                    graph_test, graph_pred = clean_graphs(graph_test, graph_pred)
                else:
                    shuffle(graph_pred)
                    graph_pred = graph_pred[0:len(graph_test)]
                print('len graph_test', len(graph_test))
                print('len graph_validate', len(graph_validate))
                print('len graph_pred', len(graph_pred))
                graph_pred_aver = 0
                for graph in graph_pred:
                    graph_pred_aver += graph.number_of_nodes()
                graph_pred_aver /= len(graph_pred)
                print('pred average len', graph_pred_aver)
                # evaluate MMD test
                mmd_degree = eval.stats.degree_stats(graph_test, graph_pred)
                mmd_clustering = eval.stats.clustering_stats(graph_test, graph_pred)
                try:
                    mmd_4orbits = eval.stats.orbit_stats_all(graph_test, graph_pred)
                except:
                    mmd_4orbits = -1
                # evaluate MMD validate
                # mmd_4orbits = eval.stats.orbit_stats_all(graph_test, graph_pred)

                mmd_degree_validate = eval.stats.degree_stats(graph_validate, graph_pred)
                mmd_clustering_validate = eval.stats.clustering_stats(graph_validate, graph_pred)
                try:
                    mmd_4orbits_validate = eval.stats.orbit_stats_all(graph_validate, graph_pred)
                except:
                    mmd_4orbits_validate = -1
                # write results
                # mmd_4orbits_validate = eval.stats.orbit_stats_all(graph_validate, graph_pred)

                f.write(
                        str(epoch)+','+
                        str(mmd_degree_validate)+','+
                        str(mmd_clustering_validate)+','+
                        str(mmd_4orbits_validate)+','+ 
                        str(mmd_degree)+','+
                        str(mmd_clustering)+','+
                        str(mmd_4orbits)+'\n')
                print('degree',mmd_degree,'clustering',mmd_clustering,'orbits',mmd_4orbits)

        # get internal MMD (MMD between ground truth validation and test sets)
        if model_name == 'Internal':
            mmd_degree_validate = eval.stats.degree_stats(graph_test, graph_validate)
            mmd_clustering_validate = eval.stats.clustering_stats(graph_test, graph_validate)
            try:
                mmd_4orbits_validate = eval.stats.orbit_stats_all(graph_test, graph_validate)
            except:
                mmd_4orbits_validate = -1
            f.write(str(-1) + ',' + str(mmd_degree_validate) + ',' + str(
                mmd_clustering_validate) + ',' + str(mmd_4orbits_validate)
                    + ',' + str(-1) + ',' + str(-1) + ',' + str(-1) + '\n')


        # get MMD between ground truth and its perturbed graphs
        if model_name == 'Noise':
            graph_validate_perturbed = perturb(graph_validate, 0.05)
            mmd_degree_validate = eval.stats.degree_stats(graph_test, graph_validate_perturbed)
            mmd_clustering_validate = eval.stats.clustering_stats(graph_test, graph_validate_perturbed)
            try:
                mmd_4orbits_validate = eval.stats.orbit_stats_all(graph_test, graph_validate_perturbed)
            except:
                mmd_4orbits_validate = -1
            f.write(str(-1) + ',' + str(mmd_degree_validate) + ',' + str(
                mmd_clustering_validate) + ',' + str(mmd_4orbits_validate)
                    + ',' + str(-1) + ',' + str(-1) + ',' + str(-1) + '\n')

        # get E-R MMD
        if model_name == 'E-R':
            graph_pred = Graph_generator_baseline(graph_train,generator='Gnp')
            # clean graphs
            if is_clean:
                graph_test, graph_pred = clean_graphs(graph_test, graph_pred)
            print('len graph_test', len(graph_test))
            print('len graph_pred', len(graph_pred))
            mmd_degree = eval.stats.degree_stats(graph_test, graph_pred)
            mmd_clustering = eval.stats.clustering_stats(graph_test, graph_pred)
            try:
                mmd_4orbits_validate = eval.stats.orbit_stats_all(graph_test, graph_pred)
            except:
                mmd_4orbits_validate = -1
            f.write(str(-1) + ',' + str(-1) + ',' + str(-1) + ',' + str(-1)
                    + ',' + str(mmd_degree) + ',' + str(mmd_clustering) + ',' + str(mmd_4orbits_validate) + '\n')


        # get B-A MMD
        if model_name == 'B-A':
            graph_pred = Graph_generator_baseline(graph_train, generator='BA')
            # clean graphs
            if is_clean:
                graph_test, graph_pred = clean_graphs(graph_test, graph_pred)
            print('len graph_test', len(graph_test))
            print('len graph_pred', len(graph_pred))
            mmd_degree = eval.stats.degree_stats(graph_test, graph_pred)
            mmd_clustering = eval.stats.clustering_stats(graph_test, graph_pred)
            try:
                mmd_4orbits_validate = eval.stats.orbit_stats_all(graph_test, graph_pred)
            except:
                mmd_4orbits_validate = -1
            f.write(str(-1) + ',' + str(-1) + ',' + str(-1) + ',' + str(-1)
                    + ',' + str(mmd_degree) + ',' + str(mmd_clustering) + ',' + str(mmd_4orbits_validate) + '\n')

        # get performance for baseline approaches
        if 'Baseline' in model_name:
            # read test graph
            for epoch in range(epoch_start, epoch_end, epoch_step):
                # get filename
                fname_pred = args.graph_save_path + model_name + '_' + dataset_name + '_' + str(
                    64) + '_pred_' + str(epoch) + '.dat'
                # load graphs
                try:
                    graph_pred = load_graph_list(fname_pred, is_real=False)  # default False
                except:
                    print('Not found: ' + fname_pred)
                    continue
                # clean graphs
                if is_clean:
                    graph_test, graph_pred = clean_graphs(graph_test, graph_pred)
                else:
                    shuffle(graph_pred)
                    graph_pred = graph_pred[0:len(graph_test)]
                print('len graph_test', len(graph_test))
                print('len graph_validate', len(graph_validate))
                print('len graph_pred', len(graph_pred))

                graph_pred_aver = 0
                for graph in graph_pred:
                    graph_pred_aver += graph.number_of_nodes()
                graph_pred_aver /= len(graph_pred)
                print('pred average len', graph_pred_aver)

                # evaluate MMD test
                mmd_degree = eval.stats.degree_stats(graph_test, graph_pred)
                mmd_clustering = eval.stats.clustering_stats(graph_test, graph_pred)
                try:
                    mmd_4orbits = eval.stats.orbit_stats_all(graph_test, graph_pred)
                except:
                    mmd_4orbits = -1
                # evaluate MMD validate
                mmd_degree_validate = eval.stats.degree_stats(graph_validate, graph_pred)
                mmd_clustering_validate = eval.stats.clustering_stats(graph_validate, graph_pred)
                try:
                    mmd_4orbits_validate = eval.stats.orbit_stats_all(graph_validate, graph_pred)
                except:
                    mmd_4orbits_validate = -1
                # write results
                f.write(str(epoch) + ',' + str(mmd_degree_validate) + ',' + str(
                    mmd_clustering_validate) + ',' + str(mmd_4orbits_validate)
                        + ',' + str(mmd_degree) + ',' + str(mmd_clustering) + ',' + str(mmd_4orbits) + '\n')
                print('degree', mmd_degree, 'clustering', mmd_clustering, 'orbits', mmd_4orbits)



        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Discrete Flow')
    parser.add_argument('--input-dir', type=str, default='/cmlscratch/kong/records/flow')
    parser.add_argument('--output-dir', type=str, default='/cmlscratch/kong/records/flow')
    parser.add_argument('--run-name', type=str, default='grid0')
    parser.add_argument('--test_file', type=str, default='', help='directly import list of test_graphs')
    parser.add_argument('--train_file', type=str, default='', help='directly import list of train_graphs')
    parser.add_argument('--epoch_start', type=int, default=50)
    parser.add_argument('--epoch_end', type=int, default=80)
    parser.add_argument('--epoch_step', type=int, default=10)
    args = parser.parse_args()
    setattr(args, 'graph_save_path', args.input_dir + '/graphs')

    models_eval = ['GraphRNN_RNN'] # a list of models to be evaluate
    datasets_eval = ['grid'] # a list of datasets trained with

    for model_name in models_eval:
        for dataset_name in datasets_eval:
            # check output exist
            fname_output = args.output_dir + model_name+'_'+ dataset_name +'.csv'
            print('processing: '+ args.output_dir + model_name + '_' + dataset_name + '.csv')
            evaluate(fname_output, model_name, dataset_name, args, 
                    epoch_start=args.epoch_start,epoch_end=args.epoch_end,epoch_step=args.epoch_step)





    
    

    
    
    


    