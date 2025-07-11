#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:21:42 2020

@author: hsjomaa
"""

import torch
import pandas as pd
import random
import numpy as np
np.random.seed(318)
random.seed(3718)
torch.manual_seed(0)

class Batch(object):
    
    def __init__(self,batch_size,fixed_shape = True):
        
        self.batch_size = batch_size
        self.fixed_shape = fixed_shape
        self.clear()
    
    def clear(self):
        # flattened triplets
        self.x = []
        # number of instances per item in triplets
        self.instances = []
        # number of features per item in triplets
        self.features = []
        # number of classes per item in triplets
        self.classes = []
        # model input
        self.input = None
        
    def append(self,instance):
        
        if len(self.x)==self.batch_size:
            
            self.clear()
            
        self.x.append(instance[0])
        self.instances.append(instance[1])
        self.features.append(instance[2])
        self.classes.append(instance[3])
        
    def collect(self):
        
        if len(self.x)!= self.batch_size and self.fixed_shape:
            raise Exception(f'Batch formation incomplete!\n{len(self.x)}!={self.batch_size}')
        
        # Convert numpy arrays to tensors before concatenating
        self.input = (torch.cat([torch.from_numpy(i) for i in self.x], axis=0).float(),
                      torch.tensor(self.classes).flatten().long(),
                      torch.tensor(self.features).flatten().long(),
                      torch.tensor(self.instances).flatten().long(),
                      )
        self.output = {'similaritytarget':torch.cat([torch.ones(self.batch_size),torch.zeros(self.batch_size)],axis=0)}

def pool(n,ntotal,shuffle):
    _pool = [_ for _ in list(range(ntotal)) if _!= n]
    if shuffle:
        random.shuffle(_pool)
    return _pool

class Sampling(object):
    def __init__(self,dataset):
        self.dataset          = dataset
        self.distribution     = pd.DataFrame(data=None,columns=['targetdataset','sourcedataset'])
        self.targetdataset   = None
        self.sourcedataset   = None

    def sample(self,batch,split,sourcesplit):
        
        nsource  = len(self.dataset.orig_data[sourcesplit]) # number of source datasets 
        ntarget  = len(self.dataset.orig_data[split]) # number of target datasets


        # print(nsource, "nsource") #80
        # print(ntarget, "ntarget") #80

        targetdataset = np.random.choice(ntarget,batch.batch_size) 

        #print the list of sampled dataset names in the targetdataset, not the indexes but the actual names
        #print([self.dataset.orig_files[split][i] for i in targetdataset], "targetdataset")
        


        batch.clear() 
        # find the negative dataset list of batch_size
        sourcedataset = []
        for target in targetdataset:
            if split==sourcesplit:
                swimmingpool  = pool(target,nsource,shuffle=True)  
            else:
                swimmingpool  = pool(-1,nsource,shuffle=True)
            sourcedataset.append(np.random.choice(swimmingpool))
        sourcedataset = np.asarray(sourcedataset).reshape(-1,)
        for target,source in zip(targetdataset,sourcedataset):
            # build instance
            instance = self.dataset.instances(target,source,split=split,sourcesplit=sourcesplit)
            batch.append(instance)
        
        distribution      = np.concatenate([targetdataset.reshape(-1,1),sourcedataset[:,None]],axis=1)
        self.distribution = pd.concat([self.distribution,\
                                       pd.DataFrame(distribution,columns=['targetdataset','sourcedataset'])],axis=0,ignore_index=True)
            
        self.targetdataset   = targetdataset  
        self.sourcedataset   = sourcedataset
        return batch
    
class TestSampling(object):
    def __init__(self,dataset):
        self.dataset          = dataset
        self.distribution     = pd.DataFrame(data=None,columns=['targetdataset','sourcedataset'])

    def sample(self,batch,split,sourcesplit,targetdataset):
        
        nsource  = len(self.dataset.orig_data[sourcesplit])
        # clear batch
        batch.clear() 
        # find the negative dataset list of batch_size
        swimmingpool  = pool(targetdataset,nsource,shuffle=True) if split==sourcesplit else pool(-1,nsource,shuffle=True)
        # double check divisibilty by batch size
        sourcedataset = np.random.choice(swimmingpool,batch.batch_size,replace=False)
        # iterate over batch negative datasets
        for source in sourcedataset:
            # build instance
            instance = self.dataset.instances(targetdataset,source,split=split,sourcesplit=sourcesplit)
            batch.append(instance)
        
        distribution      = np.concatenate([np.asarray(batch.batch_size*[targetdataset])[:,None],sourcedataset[:,None]],axis=1)
        self.distribution = pd.concat([self.distribution,\
                                       pd.DataFrame(distribution,columns=['targetdataset','sourcedataset'])],axis=0,ignore_index=True)    
            
        return batch

    def sample_from_one_dataset(self,batch):
        
        # clear batch
        batch.clear() 
        # iterate over batch negative datasets
        for _ in range(batch.batch_size):
            # build instance
            instance = self.dataset.instances()
            batch.append(instance)
        
        return batch
    