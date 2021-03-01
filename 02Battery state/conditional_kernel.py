# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:56:20 2020

@author: Qinglu Meng
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

def MLcon_kernel(X_p_list , Y_p , X_q_list , Y_q , lamda = 1):
    '''
    dim(X_p_list) = dim(X_q_list) = layer_num*Size*n_feature
    here we set layer_num = 1
    '''
    layer_num = 1
    out = 0
    for i in range(layer_num):
        X_p = X_p_list[i]
        X_q = X_q_list[i]
        np = X_p.shape[0]
        nq = X_q.shape[0]
        I1 =torch.eye(np)
        I2 =torch.eye(nq)
        Kxpxp = kernel(X_p , X_p)
        Kxqxq = kernel(X_q , X_q)
        Kxqxp = kernel(X_q , X_p)
        Kypyq = kernel(Y_p , Y_q)
        Kyqyq = kernel(Y_q , Y_q)
        Kypyp = kernel(Y_p , Y_p)
        a = torch.mm((torch.inverse(Kxpxp+np*lamda*I1)),Kypyp)
        b = torch.mm(a,(torch.inverse(Kxpxp+np*lamda*I1)))
        c = torch.mm(b,Kxpxp)
        out1 = torch.trace(c)
        
        a1 = torch.mm((torch.inverse(Kxqxq+nq*lamda*I2)),Kyqyq)
        b1 = torch.mm(a1,(torch.inverse(Kxqxq+nq*lamda*I2)))
        c1 = torch.mm(b1,Kxqxq)
        out2 = torch.trace(c1)
        
        a2 = torch.mm((torch.inverse(Kxpxp+np*lamda*I1)),Kypyq)
        b2 = torch.mm(a2,(torch.inverse(Kxqxq+nq*lamda*I2)))
        c2 = torch.mm(b2,Kxqxp)
        out3 = torch.trace(c2)       
        out += (out1 + out2 - 2*out3)        
    return out
