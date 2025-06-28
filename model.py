#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:58:58 2020

@author: hsjomaa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json
import time
import os
from modules import FunctionF,FunctionH,FunctionG,PoolF,PoolG,PoolH, get_units
torch.manual_seed(0)


class Model(nn.Module):
    '''
    Model
    '''
    def __init__(self,configuration,rootdir,for_eval=False,fine_tuning=False):
        super().__init__()
        # data shape
        self.batch_size    = configuration['batch_size']
        self.split         = configuration['split']
        self.searchspace   = configuration['searchspace']
        
        self.nonlinearity_d2v  = configuration['nonlinearity_d2v']
        # Function F
        self.units_f     = configuration['units_f']
        self.nhidden_f   = configuration['nhidden_f']
        self.architecture_f = configuration['architecture_f']
        self.resblocks_f = configuration['resblocks_f']

        # Function G
        self.units_g     = configuration['units_g']
        self.nhidden_g   = configuration['nhidden_g']
        self.architecture_g = configuration['architecture_g']
        
        # Function H
        self.units_h     = configuration['units_h']
        self.nhidden_h   = configuration['nhidden_h']
        self.architecture_h = configuration['architecture_h']
        self.resblocks_h   = configuration['resblocks_h']

        self.delta = configuration['delta']
        self.gamma = configuration['gamma']

        self.config_num = configuration["number"]
        
        self.dataset2vecmodel(trainable=True)
        self.trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)

        configuration["trainable"] = self.trainable_count

        # tracking
        self.metrickeys = ['similarityloss','time',"roc"]
        self.with_csv   = True
        # create a location if not evaluation model
        if not for_eval:
            self.directory = self._create_dir(rootdir)
            self._save_configuration(configuration)
            
    def _create_dir(self,rootdir):
        import datetime
        # create directory
        directory = os.path.join(rootdir, "checkpoints",f"searchspace-{self.searchspace}",f"split-{self.split}","dataset2vec",\
                                 "vanilla",f"configuration-{self.config_num}",datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f'))
        os.makedirs(directory, exist_ok=True)
        return directory
    
    def similarity(self,layer,positive_pair): #layer is the metafeatures
        return torch.exp(-self.gamma*self.distance(layer,positive_pair)) #similarity is the exponential of the negative distance
    
    def distance(self,layer,positive_pair):
        layer   = layer.view(self.batch_size,3,self.units_h) #layer is the metafeatures
        if not positive_pair: 
            pos = torch.mean(layer[:,:2],axis=1).unsqueeze(1)
            neg = layer[:,-1].unsqueeze(1)
        else:
            pos = layer[:,0].unsqueeze(1)
            neg = layer[:,1].unsqueeze(1)
        
        dist = torch.norm(pos - neg, p=2, dim=2)
        return dist.squeeze(1)
    
    def similarityloss(self,target_y,predicted_y):  #target_y is the similarity score, predicted_y is the metafeatures
        negative_prob   = self.similarity(predicted_y,positive_pair=False) #similarity of target train and target valid
        positive_prob   = self.similarity(predicted_y,positive_pair=True) #similarity of target train and source train
        
        logits = torch.cat([positive_prob,negative_prob],axis=0)
        similarityweight = torch.cat([torch.ones(self.batch_size),self.delta*(torch.ones(self.batch_size))],axis=0).to(logits.device)
        
        return F.binary_cross_entropy(logits, target_y, weight=similarityweight)
       
    def getmetafeatures(self,x):
        output = self.forward(x)
        layer = self.pool_h(output['metafeatures'], ignore_negative=True)
        return layer
    
    def predict(self,x,y):
        output = self.forward(x)
        phi     = output['metafeatures']
        posprob = self.similarity(phi,positive_pair=True)
        negprob = self.similarity(phi,positive_pair=False)        
        proba  = torch.cat([posprob,negprob],axis=0)
        return proba,y["similaritytarget"]
        
    def forward(self, x):

        x_in, nclasses, nfeature, ninstanc = x
        layer    = self.function_f(x_in)
        layer    = self.pool_f(layer,nclasses,nfeature,ninstanc)
        layer    = self.function_g(layer)
        layer    = self.pool_g(layer,nclasses,nfeature)
        metafeatures = self.function_h(layer)
        return {'metafeatures':metafeatures}

    def dataset2vecmodel(self,trainable):
        # Function F
        self.function_f = FunctionF(in_features=2, units=self.units_f, nhidden=self.nhidden_f, nonlinearity=self.nonlinearity_d2v, architecture=self.architecture_f, resblocks=self.resblocks_f, trainable=trainable)
        pool_f_out_dim = get_units(self.nhidden_f - 1, self.units_f, self.architecture_f, self.nhidden_f)
        self.pool_f = PoolF(units=pool_f_out_dim)
        
        # Function G
        self.function_g = FunctionG(in_features=pool_f_out_dim, units=self.units_g, nhidden=self.nhidden_g, nonlinearity=self.nonlinearity_d2v, architecture=self.architecture_g, trainable=trainable)
        pool_g_out_dim = get_units(self.nhidden_g - 1, self.units_g, self.architecture_g, self.nhidden_g)
        self.pool_g = PoolG(units=pool_g_out_dim)
        
        # Function H
        self.function_h = FunctionH(in_features=pool_g_out_dim, units=self.units_h, nhidden=self.nhidden_h, nonlinearity=self.nonlinearity_d2v, architecture=self.architecture_h, trainable=trainable, resblocks=self.resblocks_h)
        self.pool_h = PoolH(self.batch_size, self.units_h)

    def _save_configuration(self,configuration):
        configuration.update({"savedir":self.directory})
        filepath = os.path.join(self.directory,"configuration.txt")
        with open(filepath, 'w') as json_file:
          json.dump(configuration, json_file)   