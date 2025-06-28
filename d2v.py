#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 00:16:56 2020

@author: hsjomaa
"""
#from socket import RDS_RDMA_INVALIDATE
#from this import d
from dataset import Dataset
from sampling import Batch,Sampling,TestSampling
import torch
import torch.optim as optim
import copy
import json
from model import Model
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score
import os
import pandas as pd

# set random seeds
torch.manual_seed(0)
np.random.seed(42)
#---------------------------------
# create parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--configuration', help='Select model configuration', type=int,default=0)
parser.add_argument('--split', help='Select training fold', type=int,default=0)
parser.add_argument('--searchspace', help='Select metadataset',choices=['a','b','c'], type=str,default='a')
parser.add_argument('--learning_rate', help='Learning rate',type=float,default=1e-3)
parser.add_argument('--delta', help='negative datasets weight',type=float,default=2)
parser.add_argument('--gamma', help='distance hyperparameter',type=float,default=1)
parser.add_argument('--device', help='Device to use for training (e.g., "cpu" or "cuda:0")', type=str, default='cpu')



# Default: split 0, searchspace a, learning rate 1e-3, delta 2, gamma 1, device cpu

args    = parser.parse_args()

rootdir     = os.path.dirname(os.path.realpath(__file__))
config_file = os.path.join(rootdir, "configurations",f"configuration-{args.configuration}.json")
info_file   = os.path.join(rootdir, "metadataset"  ,"info.json")
# load configuration
configuration = json.load(open(config_file,'r'))
# update with shared configurations with specifics
config_specs = {
    'split':	args.split,
    'searchspace':	args.searchspace,
    'learning_rate':	args.learning_rate,
    'delta':	args.delta,
    'gamma':	args.gamma,
    'minmax':	True,    
    'batch_size':	16,
    'input_shape': 2,
    }

configuration.update(config_specs)

searchspaceinfo = json.load(open(info_file,'r'))
configuration.update(searchspaceinfo[args.searchspace])
device = torch.device(args.device)

# create Dataset
normalized_dataset = Dataset(configuration,rootdir,use_valid=True)

# load training sets
nsource = len(normalized_dataset.orig_data['train'])
ntarget = len(normalized_dataset.orig_data['valid'])
ntest   = len(normalized_dataset.orig_data['test'])

# create model
model     = Model(configuration,rootdir=rootdir).to(device)
optimizer = optim.Adam(model.parameters(), lr=configuration['learning_rate'])
batch     = Batch(configuration['batch_size'])

# Define training parameters
epochs = 10000 # 10000 epochs
sampler     = Sampling(dataset=normalized_dataset)
testsampler = TestSampling(dataset=normalized_dataset)




"""

Wiki for understanding the TRAINING code:
    say batch = 64
    inputs = (64 + 64 + 64) datasets (target train, source train, target valid)
    target train and target valid are the same dataset
    target train and source train are disjoint sets of datasets (no overlap)
    thus outputs = (64 + 64 + 64) metafeatures (target train, source train, target valid) of dimesnion 32 -> (64x3, 32)
    and target (the similarity score) is of shape (64x2) -> (64 scores (1) for target train and target valid pair)
    + 64 scores (0) for source train and target valid pair

"""



base_training = True
epochs = 500


fine_tuning = True
epochs_ft = 10000


######################################
# TRAINING CODE USING AUXILIARY LOSS
######################################


early = 0
best_auc = -np.inf
verbose = False


if base_training:
    for epoch in range(epochs):

        
        model.train()
        batch = sampler.sample(batch,split='train',sourcesplit='train')
        batch.collect() #collect the batch


        inputs = [d.to(device) for d in batch.input] 


        targets = batch.output['similaritytarget'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        metafeatures = outputs['metafeatures']

        #check if metafeatures has nan values
        if torch.isnan(metafeatures).any():
            print("metafeatures has nan values")
            exit()
        
        loss = model.similarityloss(targets, metafeatures)

        #outputs is a dictionary with keys 'metafeatures'

        if verbose:
            print("--------------------------------")
            print(sampler.targetdataset, "sampler.targetdataset")
            print(batch.batch_size, "batch size")
            print(outputs.keys(), "outputs keys")
            print(outputs['metafeatures'].shape, "outputs metafeatures shape")
            print(targets, "targets")
            print(epoch,"<-epoch" , loss, "<-loss", metafeatures.shape, "<-metafeatures shape", targets.shape, "<-targets shape", inputs[0].shape, "<-inputs shape")
            print("--------------------------------")

        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        print(epoch,"<-epoch" , loss, "<-loss")

        if epoch % 50 == 0:
            model.eval()
            y_true = []
            y_pred = []
            with torch.no_grad():
                for _ in range(ntarget):
                    for q in range(10):
                        test_batch = testsampler.sample(batch,split='valid',sourcesplit='train',targetdataset=_)
                        test_batch.collect()
                        t_inputs = [d.to(device) for d in test_batch.input]
                        t_outputs = {k: v.to(device) for k,v in test_batch.output.items()}

                        prob,label = model.predict(t_inputs, t_outputs)
                        y_pred.append(prob.cpu().numpy())
                        y_true.append(label.cpu().numpy())

            y_true = np.hstack(y_true)
            y_pred = np.hstack(y_pred)
            auc_score = roc_auc_score(y_true, y_pred)
            print(f"Epoch {epoch}, Loss: {loss.item()}, Validation AUC: {auc_score}")
            
            if np.abs(auc_score - best_auc) > 1e-3:
                best_auc = auc_score
                early = 0
                torch.save(model.state_dict(), os.path.join(model.directory, "model.pth"))
            else:
                early +=1

        sampler_file = os.path.join(model.directory,"distribution.csv")
        sampler.distribution.to_csv(sampler_file)
        testsampler.distribution.to_csv(os.path.join(model.directory,"valid-distribution.csv"))
        if early > 16:
            break

######################################
# FINE TUNING CODE USING FROBENIUOUS METRIC
######################################

sanity_check = True

def pairwise_distances_squared(A):    # NOT NUMERICALLY STABLE
    norms = (A ** 2).sum(dim=1, keepdim=True)

    c = norms + norms.T - 2 * (A @ A.T)

    #check if c has any NEGATIVE values
    if c.min() < 0:
        print("pairwise_distances_squared has negative values")
        exit()
    else:
        print("pairwise_distances_squared does not have negative values")

    return norms + norms.T - 2 * (A @ A.T)


data = pd.read_csv("ft_dataset/classifiers_performance.csv")
print("fine tuning dataset loaded")

batch_size_ft = 32 #batch size for fine tuning
batch_ft    = Batch(batch_size_ft)


for epoch_ft in range(epochs_ft):


    model.train()
    batch_ft = sampler.sample(batch_ft,split='train',sourcesplit='train')
    batch_ft.collect()



    target_datasets = sampler.targetdataset
    #print the list of sampled dataset names in the targetdataset, not the indexes but the actual names
    #print([normalized_dataset.orig_files['train'][i] for i in target_datasets], "target datasets")


    inputs = [d.to(device) for d in batch_ft.input] 


    optimizer.zero_grad()

    outputs = model(inputs)
    metafeatures = outputs['metafeatures']

    #check if metafeatures has nan values
    # if torch.isnan(metafeatures).any():
    #     print("metafeatures has nan values")
    #     exit()
    # else:
    #     print("metafeatures does not have nan values")

    #set of unique metafeatures
    mfs_1 = metafeatures[:batch_size_ft, :] # Unique datasets
    mfs_2 = metafeatures[batch_size_ft:batch_size_ft*2, :] #representation of same dataset with different samped data

    sampled_datasets = [normalized_dataset.orig_files['train'][i] for i in target_datasets]
    filtered_datasets = []

    # Initialize filtered metafeatures tensors
    mfs_1_new = []
    mfs_2_new = []

    for i in range(len(sampled_datasets)):
        dataset_name = sampled_datasets[i]
        dataset_mfs = mfs_1[i:i+1, :]  # Keep as tensor with shape (1, feature_dim)
        dataset_mfs_2 = mfs_2[i:i+1, :]  # Keep as tensor with shape (1, feature_dim)
        #check if the dataset_name is in the data dataframe
        if dataset_name in data['Dataset'].values:
            filtered_datasets.append(dataset_name)
            mfs_1_new.append(dataset_mfs)
            mfs_2_new.append(dataset_mfs_2)
    
    # Convert lists to tensors if any datasets were found
    if mfs_1_new:
        mfs_1_new = torch.cat(mfs_1_new, dim=0)  # Concatenate along batch dimension
        mfs_2_new = torch.cat(mfs_2_new, dim=0)  # Concatenate along batch dimension
    else:
        # Handle case where no datasets are in the performance data
        print("Warning: No datasets from batch found in performance data")
        mfs_1_new = torch.empty((0, mfs_1.shape[1]), dtype=mfs_1.dtype, device=mfs_1.device)
        mfs_2_new = torch.empty((0, mfs_2.shape[1]), dtype=mfs_2.dtype, device=mfs_2.device)

    """
    Wiki for understanding the FINE TUNING code:
    """

    loss = torch.zeros(1, device=device)


    X1 =  mfs_1_new # N, 32
    X2 =  mfs_2_new # N, 32


    #check if X1 has nan values
    if torch.isnan(X1).any():
        print("X1 has nan values")
        exit()
    if torch.isnan(X2).any():
        print("X2 has nan values")
        exit()


    # R = torch.stack([
    # torch.tensor(
    #     data[data['Dataset'] == name][['DT', 'RF', 'SVM', 'NB', 'KNN']].values[0],
    #     dtype=torch.float32,
    #     device=device
    # )
    # for name in filtered_datasets
    # ])  # N, 5


    # dd_matrix = pairwise_distances_squared(X1)


    # ######  OPTIONAL ########
    # diag_dist = ((X1 - X2) ** 2).sum(dim=1)  # shape: (N,)
    # dd_matrix.fill_diagonal_(0)              # Clear diagonal
    # dd_matrix += torch.diag(diag_dist)       # Insert custom diagonal
    # ##########################

    # #check if dd_matrix has nan values
    # if torch.isnan(dd_matrix).any():
    #     print("dd_matrix has nan values")
    #     exit()
    # else:
    #     print("dd_matrix does not have nan values")
    # if torch.isnan(R).any():
    #     print("R has nan values")
    #     exit()
    # else:
    #     print("R does not have nan values")

    # dr_matrix = pairwise_distances_squared(R)

    # if torch.isnan(dr_matrix).any():
    #     print("dr_matrix has nan values")
    #     exit()
    # else:
    #     print("dr_matrix does not have nan values")

    # #print max and min of dd_matrix and dr_matrix
    # print("max of dd_matrix", dd_matrix.max(), "min of dd_matrix", dd_matrix.min())
    # print("max of dr_matrix", dr_matrix.max(), "min of dr_matrix", dr_matrix.min())

    # loss = ((dd_matrix.sqrt() - dr_matrix.sqrt()) ** 2).sum()

    # BAD APPROACH BUT NOT NUMERICALLY STABLE
    for i in range(len(filtered_datasets)):
        for j in range(len(filtered_datasets)):
            if i != j:
                dd = torch.norm(mfs_1_new[i] - mfs_1_new[j], p=2)
                rs_i = torch.tensor(data[data['Dataset'] == filtered_datasets[i]][['DT', 'RF', 'SVM', 'NB', 'KNN']].values,dtype=torch.float32,device=device)
                rs_j = torch.tensor(data[data['Dataset'] == filtered_datasets[j]][['DT', 'RF', 'SVM', 'NB', 'KNN']].values,dtype=torch.float32,device=device)
                dr = torch.norm(rs_i - rs_j, p=2)
                loss += (dd - dr)**2
            else: #i == j
                dd = torch.norm(mfs_1_new[i] - mfs_2_new[j], p=2)
                rs_i = torch.tensor(data[data['Dataset'] == filtered_datasets[i]][['DT', 'RF', 'SVM', 'NB', 'KNN']].values,dtype=torch.float32,device=device)
                rs_j = torch.tensor(data[data['Dataset'] == filtered_datasets[j]][['DT', 'RF', 'SVM', 'NB', 'KNN']].values,dtype=torch.float32,device=device)
                dr = torch.norm(rs_i - rs_j, p=2)
                loss += (dd - dr)**2

    print(len(filtered_datasets), "number of filtered datasets")

    loss = loss / (len(filtered_datasets) * (len(filtered_datasets)))


    #check if loss has nan values
    if torch.isnan(loss):
        print("loss has nan values")
        exit()
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()    

    print("loss", loss, "epoch", epoch_ft)


######################################
######################################

torch.save(model.state_dict(), os.path.join(model.directory, "model.pth"))

metafeatures_df = pd.DataFrame(data=None)
splitmf = []
filesmf = []
model.eval()
with torch.no_grad():
    for splits in [("train",nsource),("valid",ntarget),("test",ntest)]:
        for i in range(splits[1]):
            datasetmf = []
            for q in range(10):
                batch = testsampler.sample(batch,split=splits[0],sourcesplit='train',targetdataset=i)
                batch.collect()
                b_inputs = [d.to(device) for d in batch.input]
                datasetmf.append(model.getmetafeatures(b_inputs).cpu().numpy())
            splitmf.append(np.vstack(datasetmf).mean(axis=0)[None])
        filesmf +=normalized_dataset.orig_files[splits[0]]
splitmf = np.vstack(splitmf)
metafeatures_df = pd.DataFrame(data=splitmf,index=filesmf)
metafeatures_df.to_csv(os.path.join(model.directory,"meta-feautures.csv"))