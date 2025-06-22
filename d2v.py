#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 00:16:56 2020

@author: hsjomaa
"""
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
epochs = 10000
sampler     = Sampling(dataset=normalized_dataset)
testsampler = TestSampling(dataset=normalized_dataset)

early = 0
best_auc = -np.inf
for epoch in range(epochs):
    print(f"Epoch {epoch}")
    
    model.train()
    batch = sampler.sample(batch,split='train',sourcesplit='train')
    batch.collect()

    inputs = [d.to(device) for d in batch.input]
    targets = batch.output['similaritytarget'].to(device)
    
    optimizer.zero_grad()
    
    outputs = model(inputs)
    metafeatures = outputs['metafeatures']
    
    loss = model.similarityloss(targets, metafeatures)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()

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