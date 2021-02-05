# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 18:59:45 2020

@author: MXM
"""

#定义源域和目标域模型，并训练源域模型
import numpy as np
import torch
import Loss_evalute
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt


class SourceCNN(nn.Module):
    def  __init__(self):
        super(SourceCNN,self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(1,64,5,stride=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,64,3,stride=3),
            nn.MaxPool1d(2, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU()
            )
        self.fc1 = nn.Linear(768,384)
        self.fc2 = nn.Linear(384,64)
        self.fc3 = nn.Linear(64, 16)
        self.predict = nn.Linear(16,1)
        self.relu = nn.ReLU()
                
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        mmd_loss = self.relu(x)
        x = self.fc2(mmd_loss)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)         
        result = self.predict(x)
        
        return result, mmd_loss
    
class TargetCNN(nn.Module):
    def  __init__(self):
        super(TargetCNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1,64,5,stride=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,64,3,stride=3),
            nn.MaxPool1d(2, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU()
            )
        

        self.fc1 = nn.Linear(768,384)
        self.fc2 = nn.Linear(384,64)
        self.fc3 = nn.Linear(64, 16)
        self.predict = nn.Linear(16,1)
        self.relu = nn.ReLU()
                
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        inter_x1 = self.relu(x)
        x = self.fc2(inter_x1)
        inter_x2 = self.relu(x)
        x = self.fc3(inter_x2)
        inter_x3 = self.relu(x)         
        result = self.predict(inter_x3)
        
        target_list = list([inter_x1, inter_x2 ,inter_x3])
        
        return target_list ,result
        
    
if __name__ == '__main__':
    
    #导入数据
    
    battery7 = pd.read_csv("data/7.csv", header=None).values.reshape(168,1,371)
    label7 = pd.read_csv("data/L7.csv", header = None).values
    
    battery6 = pd.read_csv("data/6.csv", header=None).values.reshape(168,1,371)
    label6 = pd.read_csv("data/L6.csv", header = None).values
    
    battery5 = pd.read_csv("data/5.csv", header=None).values.reshape(168,1,371)
    label5 = pd.read_csv("data/L5.csv", header = None).values
    
        
    index = [i for i in range(168)]
        
    source_x = torch.Tensor(battery6[index,:,:])
    source_y = torch.Tensor(label6[index,:])
    target_x = torch.Tensor(battery7)
    target_y = torch.Tensor(label7)
                                  
    Source = SourceCNN()
    learning_rate = 0.01
    regularization  = 1e-5
    num_epochs = 100
    lamda = 0.01
    optimizer = torch.optim.Adam(Source.parameters(), lr = learning_rate , weight_decay = regularization)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(num_epochs):
        sprediction, smmd_loss = Source.forward(source_x)
        tprediction, tmmd_loss = Source.forward(target_x)
        Loss = criterion(sprediction , source_y)+lamda*torch.norm((torch.mean(smmd_loss,axis=0))-torch.mean(tmmd_loss,axis=0))
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()  
        if (epoch%1==0):
            print(epoch,Loss.data)
    

    sypre = Source.forward(source_x)[0]
    typre = Source.forward(target_x)[0]
    print('拟合源域的效果')
    Loss_evalute.loss_evaluate(source_y.data.numpy(),sypre.data.numpy())
    print('在目标域上的效果')
    Loss_evalute.loss_evaluate(target_y.data.numpy(),typre.data.numpy())
            
    #画图显示
    tool = 'battery6->battery7'
    cycle = (np.arange(168)).reshape(168,1)
    figure = plt.figure()
    plt.xlabel('Cycle_'+tool,fontsize=15)
    plt.ylabel('Capacity',fontsize=15)
    plt.plot(cycle,sypre.data.numpy(),label='source-prediction',c='C9',lw=1.5)
    plt.plot(cycle,source_y.data.numpy(),label='source-True',c = 'r',lw=2)
    plt.plot(cycle, typre.data.numpy() , label = 'target-prediction',lw =1.5,c = 'g')

    
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.show()
   
    torch.save(Source.state_dict(),"pre-trained model/model.pth")
        
    
    


