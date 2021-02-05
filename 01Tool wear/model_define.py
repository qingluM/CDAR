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
import torch.utils.data as Data

class SourceCNN(nn.Module):
    def  __init__(self):
        super(SourceCNN,self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(1,64,5,stride=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Conv1d(64,64,5,stride=5),
            nn.MaxPool1d(2, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64,64,5,stride=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,64,3,stride=3),
            nn.MaxPool1d(2, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU()
            )
        self.fc1 = nn.Linear(832,384)
        self.fc2 = nn.Linear(384,64)
        self.fc3 = nn.Linear(64, 16)
        self.predict = nn.Linear(16,1)
        self.relu = nn.ReLU()
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        source = self.fc1(x)
        x = self.relu(source)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)         
        result = self.predict(x)
        
        return result
    
class TargetCNN(nn.Module):
    def  __init__(self):
        super(TargetCNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1,64,5,stride=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,64,5,stride=5),
            nn.MaxPool1d(2, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64,64,5,stride=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,64,3,stride=3),
            nn.MaxPool1d(2, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU()
            )
        self.fc1 = nn.Linear(832,384)
        self.fc2 = nn.Linear(384,64)
        self.fc3 = nn.Linear(64, 16)
        self.predict = nn.Linear(16,1)
        self.relu = nn.ReLU()
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
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
    
    c6 = np.load('data/c6.npy').reshape(313,1,20000)
    c4 = np.load('data/c4.npy').reshape(313,1,20000)
    
    Tool_42 = pd.read_csv("data/c4_wear.csv").values[2:316,2].reshape(313,1)
    Tool_62 = pd.read_csv("data/c6_wear.csv").values[2:316,2].reshape(313,1)
    
    Tool_43 = pd.read_csv("data/c4_wear.csv").values[2:316,3].reshape(313,1)
    Tool_63 = pd.read_csv("data/c6_wear.csv").values[2:316,3].reshape(313,1)
    
        
    index = [i for i in range(313)]
    
    #Set up source domain and target domain here
    source_x = torch.Tensor(c4[index,:,:])
    source_y = torch.Tensor(Tool_42[index,:])
    target_x = torch.Tensor(c6)
    target_y = torch.Tensor(Tool_62)
                                  
    Source = SourceCNN()
    learning_rate = 0.02
    regularization  = 1e-4
    num_epochs = 10#50
    Batch_size = 64 
    optimizer = torch.optim.Adam(Source.parameters(), lr = learning_rate , weight_decay = regularization)
    criterion = torch.nn.MSELoss()
    
    
    torch_dataset = Data.TensorDataset(source_x, source_y)
    loader = Data.DataLoader(
            dataset = torch_dataset,
            batch_size = Batch_size,
            shuffle=True,
            )
    
    
    for epoch in range(num_epochs):
        for step,(batch_x,batch_y) in enumerate(loader):
                prediction = Source.forward(batch_x)
                Loss = criterion(prediction , batch_y)
                optimizer.zero_grad()
                Loss.backward()
                optimizer.step()  
                if (step%1==0):
                    print(epoch,Loss.data)
    

    sypre = Source.forward(source_x)
    typre = Source.forward(target_x)
    print('results on source domain')
    Loss_evalute.loss_evaluate(source_y.data.numpy(),sypre.data.numpy())
    print('results on source domain')
    Loss_evalute.loss_evaluate(target_y.data.numpy(),typre.data.numpy())
            
    #画图显示
    tool = 'Blade2'
    cycle = (np.arange(313)).reshape(313,1)
    figure = plt.figure()
    plt.title('T4-Blade2-training',fontsize=15)
    plt.xlabel('Milling times',fontsize=13)
    plt.ylabel('Wear(μm)',fontsize=13)
    plt.plot(cycle,sypre.data.numpy(),label='source-prediction',c='C9',lw=1.8)
    plt.plot(cycle,source_y.data.numpy(),label='source-True',c = 'r',lw=3)
    plt.plot(cycle, typre.data.numpy() , label = 'target-prediction',lw =1.5,c = 'g')

    
    plt.legend(fontsize=12)
    plt.show()
   
    torch.save(Source.state_dict(),"pre-trained model/pretrained-model.pth")
        
    
    


