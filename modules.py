#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 00:02:51 2020

@author: hsjomaa
"""

import torch
import torch.nn as nn

def importance(config):
    if config['importance'] == 'linear':
        fn = lambda x:x
    elif config['importance'] == 'None':
        fn = None
    else:
        raise Exception('please define an importance function')
    return fn
    
ARCHITECTURES = ['SQU','ASC','DES','SYM','ENC']
def get_units(idx,neurons,architecture,layers=None):
    assert architecture in ARCHITECTURES
    if architecture == 'SQU':
        return neurons
    elif architecture == 'ASC':
        return (2**idx)*neurons
    elif architecture == 'DES':
        assert layers is not None, "layers must be specified for DES architecture"
        return (2**(layers-1-idx))*neurons    
    elif architecture=='SYM':
        assert (layers is not None and layers > 2)
        if layers%2==1:
            return neurons*2**(int(layers/2) - abs(int(layers/2)-idx))
        else:
            x = int(layers/2)
            idx  = idx if idx < x else 2*x-idx -1
            return neurons*2**(int(layers/2) - abs(int(layers/2)-idx))
        
    elif architecture=='ENC':
        assert (layers is not None and layers > 2)
        if layers%2==0:
            x = int(layers/2)
            idx  = idx if idx < x else 2*x-idx -1            
            return neurons*2**(int(layers/2)-1 -idx)
        else:
            x = int(layers/2)
            idx  = idx if idx < x else 2*x-idx
            return neurons*2**(int(layers/2) -idx)

def get_nonlinearity(nonlinearity):
    if nonlinearity == 'relu':
        return nn.ReLU()
    elif nonlinearity == 'tanh':
        return nn.Tanh()
    elif nonlinearity == 'sigmoid':
        return nn.Sigmoid()
    else:
        return getattr(nn, nonlinearity)()

class ResidualBlock(nn.Module):
    def __init__(self, units, nhidden, nonlinearity, architecture, trainable):
        super().__init__()
        self.n = nhidden
        self.units = units
        self.nonlinearity = get_nonlinearity(nonlinearity)
        
        layers = []
        for i in range(self.n):
            layer = nn.Linear(self.units, self.units)
            if not trainable:
                for param in layer.parameters():
                    param.requires_grad = False
            layers.append(layer)
        self.block = nn.ModuleList(layers)

    def forward(self, x):
        residual = x
        for i, layer in enumerate(self.block):
            x = layer(x)
            if i < (self.n - 1):
                x = self.nonlinearity(x)
        return self.nonlinearity(x + residual)

class FunctionF(nn.Module):
    def __init__(self, in_features, units, nhidden, nonlinearity, architecture, trainable, resblocks=0):
        super().__init__()
        self.resblocks = resblocks
        self.nonlinearity = get_nonlinearity(nonlinearity)
        
        layers = []
        if resblocks > 0:
            assert architecture == "SQU", "Residual blocks only supported for SQU architecture"
            layers.append(nn.Linear(in_features, units))
            for _ in range(resblocks):
                layers.append(ResidualBlock(units=units, nhidden=nhidden, nonlinearity=nonlinearity, architecture=architecture, trainable=trainable))
            layers.append(nn.Linear(units, units))
        else:
            current_dim = in_features
            for i in range(nhidden):
                output_dim = get_units(i, units, architecture, nhidden)
                layers.append(nn.Linear(current_dim, output_dim))
                current_dim = output_dim
        
        self.block = nn.ModuleList(layers)
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        for i, fc in enumerate(self.block):
            x = fc(x)
            if self.resblocks == 0:
                x = self.nonlinearity(x)
            else:
                if i == 0 or i == (len(self.block) - 1):
                    x = self.nonlinearity(x)
        return x

class PoolF(nn.Module):
    def __init__(self,units):    
        super(PoolF, self).__init__()
        self.units = units
        
    def forward(self,x,nclasses,nfeature,ninstanc):

        s = (nclasses * nfeature * ninstanc).view(-1).tolist()

        x_split = torch.split(x, s, dim=0)
        
        e  = []
        for i,bx in enumerate(x_split):
            te = bx.view(1,nclasses[i],nfeature[i],ninstanc[i],self.units)
            te = torch.mean(te,axis=3)
            e.append(te.view(nclasses[i]*nfeature[i],self.units))
            
        return torch.cat(e,axis=0)
    
class FunctionG(nn.Module):
    def __init__(self, in_features, units, nhidden, nonlinearity, architecture, trainable):
        super().__init__()
        self.nonlinearity = get_nonlinearity(nonlinearity)
        
        layers = []
        current_dim = in_features
        for i in range(nhidden):
            output_dim = get_units(i, units, architecture, nhidden)
            layers.append(nn.Linear(current_dim, output_dim))
            current_dim = output_dim
        
        self.block = nn.ModuleList(layers)
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        for fc in self.block:
            x = fc(x)
            x = self.nonlinearity(x)
        return x

class PoolG(nn.Module):
    def __init__(self,units):    
        super(PoolG, self).__init__()
        self.units = units
        
    def forward(self, x,nclasses,nfeature):
        s = (nclasses * nfeature).tolist()
        x_split = torch.split(x, s, dim=0)
        
        e  = []
        for i,bx in enumerate(x_split):
            te = bx.view(1,nclasses[i]*nfeature[i],self.units)
            te = torch.mean(te,axis=1)
            e.append(te)
            
        return torch.cat(e,axis=0)

class FunctionH(nn.Module):
    def __init__(self, in_features, units, nhidden, nonlinearity, architecture, trainable, resblocks=0):
        super().__init__()
        self.resblocks = resblocks
        self.nonlinearity = get_nonlinearity(nonlinearity)
        
        layers = []
        if resblocks > 0:
            assert architecture == "SQU", "Residual blocks only supported for SQU architecture"
            layers.append(nn.Linear(in_features, units))
            for _ in range(resblocks):
                layers.append(ResidualBlock(units=units, nhidden=nhidden, nonlinearity=nonlinearity, architecture=architecture, trainable=trainable))
            layers.append(nn.Linear(units, units))
        else:
            current_dim = in_features
            for i in range(nhidden):
                output_dim = get_units(i, units, architecture, nhidden)
                layers.append(nn.Linear(current_dim, output_dim))
                current_dim = output_dim
        
        self.block = nn.ModuleList(layers)
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self,x):
        for i,fc in enumerate(self.block):
            x = fc(x)
            if self.resblocks == 0:
                if i < len(self.block) - 1:
                    x = self.nonlinearity(x)
            else:
                if i==0 or i == (len(self.block)-1):
                    x = self.nonlinearity(x)
        return x

class PoolH(nn.Module):
    def __init__(self, batch_size,units):
        super(PoolH, self).__init__()
        self.batch_size = batch_size
        self.units = units
        
    def forward(self, x,ignore_negative):
        e  =  x.view(self.batch_size,3,self.units)
        e1 = torch.mean(e[:,:2],dim=1)
        if not ignore_negative:
            e1 = e[:,-1]
        e  = e1.view(self.batch_size,self.units)            
        return e