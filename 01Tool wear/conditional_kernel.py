# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:56:20 2020

@author: MXM
"""

import torch

def kernel(X, X2, gamma=0.4):
    '''
    Input: X  Size1*n_feature
           X2 Size2*n_feature
    Output: Size1*Size2
    '''
    X = torch.transpose(X,1,0)
    X2 = torch.transpose(X2,1,0)
    n1, n2 = X.shape[1],X2.shape[1]
    n1sq = torch.sum(X ** 2, 0)
    n1sq = n1sq.float()
    n2sq = torch.sum(X2 ** 2, 0)
    n2sq = n2sq.float()
    D = torch.ones((n1, n2)) * n2sq + torch.transpose((torch.ones((n2, n1)) * n1sq),1,0)+  - 2 * torch.mm(torch.transpose(X,1,0), X2)
    K = torch.exp(-gamma * D)
    
    return K

def MLcon_kernel(source_list , Y_s , target_list , Y_t , lamda = 1):
    
    '''
    dim(X_s) = layer_num*Size*n_feature
    here we set layer_num = 1
    '''
    layer_num = 1
    out = 0
    for i in range(layer_num):
        X_s = source_list[i]
        X_t = target_list[i]
        ns = X_s.shape[0]
        nt = X_t.shape[0]
        I1 =torch.eye(ns)
        I2 =torch.eye(nt)
        Kxsxs = kernel(X_s , X_s)
        Kxtxt = kernel(X_t , X_t)
        Kxtxs = kernel(X_t , X_s)
        Kysyt = kernel(Y_s , Y_t)
        Kytyt = kernel(Y_t , Y_t)
        Kysys = kernel(Y_s , Y_s)
        a = torch.mm((torch.inverse(Kxsxs+ns*lamda*I1)),Kysys)
        b = torch.mm(a,(torch.inverse(Kxsxs+ns*lamda*I1)))
        c = torch.mm(b,Kxsxs)
        out1 = torch.trace(c)
        
        a1 = torch.mm((torch.inverse(Kxtxt+nt*lamda*I2)),Kytyt)
        b1 = torch.mm(a1,(torch.inverse(Kxtxt+nt*lamda*I2)))
        c1 = torch.mm(b1,Kxtxt)
        out2 = torch.trace(c1)
        
        a2 = torch.mm((torch.inverse(Kxsxs+ns*lamda*I1)),Kysyt)
        b2 = torch.mm(a2,(torch.inverse(Kxtxt+nt*lamda*I2)))
        c2 = torch.mm(b2,Kxtxs)
        out3 = torch.trace(c2)       
        out += (out1 + out2 - 2*out3)
            
    return out
