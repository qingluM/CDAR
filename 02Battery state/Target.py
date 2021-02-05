# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 20:32:14 2020

@author: MXM
"""
import numpy as np
import torch
import Loss_evalute
import conditional_kernel
import pandas as pd
import matplotlib.pyplot as plt
import model_define
import scipy.io as sio
import JDD
import torch.utils.data as Data



def load_TargetCNN(model,name):
     
    pretrained_dict = torch.load("pre-trained model/model.pth")
    new_dict = model.state_dict()
    pretrained_dict1 = {k: v for k, v in pretrained_dict.items() if k in new_dict}
    new_dict.update(pretrained_dict1)
    model.load_state_dict(new_dict)
    
    if name=='CDK':
        namelist = ['predict.bias','predict.weight','fc3.bias','fc3.weight','fc2.bias','fc2.weight','fc1.bias','fc1.weight']
        for name, value in model.named_parameters():
            if name in namelist:
                value.requires_grad = True
            else:
                value.requires_grad = False       
    else:
        namelist = []
        for name, value in model.named_parameters():
            if name not in namelist:
                value.requires_grad = True
            else:
                value.requires_grad = False
   
    print('#######装填#######')  
        

def train_TargetCNN(model, t_xtrain, t_ytrain, epoch = 800, learning_rate = 0.0003, regularization = 1e-4):
    
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr = learning_rate, weight_decay = regularization)
    criterion = torch.nn.MSELoss()
    
    for i in range(epoch):

        InterxList, prediction = model.forward(t_xtrain)
        Loss = criterion(prediction, t_ytrain)
        optimizer.zero_grad()
        Loss.backward()        
        optimizer.step()
        if (i%10==0):
            print(i, Loss.data)


def trainagain_TargetCNN(name, model, t_xtrain, t_ytrain, s_xtrain, epoch = 800, learning_rate = 0.0003, regularization = 1e-5, gamma = 0.5, lamda = 0.25):
    
    params = filter(lambda p: p.requires_grad, model.parameters())
    
    lr = learning_rate
    fc1_params = list(map(id, model.fc1.parameters()))
    fc2_params = list(map(id, model.fc2.parameters()))
    fc3_params = list(map(id, model.fc3.parameters()))
    #这里记得加一个fc1
    predict_params = filter(lambda p :id(p) not in fc1_params + fc2_params + fc3_params, params)

    optimizer = torch.optim.Adam([
                {'params': predict_params},
                {'params': model.fc3.parameters(), 'lr':lr*0.1},
                {'params': model.fc2.parameters(), 'lr':lr*0.1},
                {'params': model.fc1.parameters(), 'lr':lr*0.1}],
                lr = learning_rate, weight_decay = regularization)
    
    criterion = torch.nn.MSELoss()
    
    for i in range(epoch):
        source_list, prediction = model.forward(s_xtrain)
        target_list, prediction2 = model.forward(t_xtrain)

        cdk_loss = conditional_kernel.MLcon_kernel(source_list, prediction, target_list, t_ytrain)
        Loss = gamma*criterion(prediction2, t_ytrain) + lamda*cdk_loss
              
        optimizer.zero_grad()
        Loss.backward()       
        optimizer.step()
        if (i%10==0):
            print(i, Loss.data)

            
def test_TargetCNN(model, t_xtest):
    
    inter_x, ypre = model.forward(t_xtest)
    for i in range(3):
        inter_x[i] = inter_x[i].data.numpy()
        
    ypre = ypre.data.numpy()
      
    return inter_x, ypre

if __name__ == '__main__':
                  
    #设置目标域数据集    
    battery7 = pd.read_csv("data/7.csv", header=None).values.reshape(168,1,371)
    label7 = pd.read_csv("data/L7.csv", header = None).values
    
    battery6 = pd.read_csv("data/6.csv", header=None).values.reshape(168,1,371)
    label6 = pd.read_csv("data/L6.csv", header = None).values
    
    battery5 = pd.read_csv("data/5.csv", header=None).values.reshape(168,1,371)
    label5 = pd.read_csv("data/L5.csv", header = None).values
    
    target_size = 10
    seed_number = 25
    SeedResult = np.zeros((seed_number,16))
    SeedRecord = np.zeros((seed_number,target_size))

    for i in range(seed_number):
                
        SEED = i
        i_str = str(i)
        np.random.seed(SEED)
        
        #随机取点
        index1 = np.random.choice(battery6.shape[0],target_size)
        index2 = np.delete(np.arange(168),index1)
        
        SeedRecord[i,:] = index1 
        
        t_xtrain = battery7[index1,:,:]
        t_ytrain = label7[index1,:]
        t_xtest = np.delete(battery7, index1, axis = 0)
        t_ytest = np.delete(label7, index1, axis = 0)
        
        s_xtrain = torch.Tensor(battery6)
        s_ytrain = torch.Tensor(label6)
        t_xtrain = torch.Tensor(t_xtrain)
        t_ytrain = torch.Tensor(t_ytrain)
        t_xtest = torch.Tensor(t_xtest)
        t_ytest = torch.Tensor(t_ytest)
        
        target1 = model_define.TargetCNN()

####1111111111111111111111111111111111仅使用目标域数据进行训练11111111111111111111111111111111111111111####        
        if 1:
            '''
            仅使用目标域数据进行训练
            '''
            name = 'OnlyTarget'
            learning_rate = 0.03
            regularization  = 1e-3
            num_epochs = 100
            
            optimizer = torch.optim.Adam(target1.parameters(), lr = learning_rate , weight_decay = regularization)
            criterion = torch.nn.MSELoss()           
            print('###############开始新的一次程序###############')
            
            for epoch in range(num_epochs):
                prediction = target1.forward(t_xtrain)[1]
                Loss = criterion(prediction , t_ytrain)
                optimizer.zero_grad()
                Loss.backward()
                optimizer.step()  
                if (epoch%20==0):
                    print(epoch,Loss.data)
                    
            '''
            ot----only target
            '''
            ot_Xtest, ot_Ytest_pre = test_TargetCNN(target1, t_xtest)
            print('第',i,'个种子时仅使用目标域数据的效果：')
            result1 = Loss_evalute.loss_evaluate(t_ytest.data.numpy(),ot_Ytest_pre)
            ot_Xtrain, ot_Ytrain_pre = test_TargetCNN(target1, t_xtrain)      
            

####22222222222222222222222222222222222仅使用源域进行训练22222222222222222222222222222222222#######
        if 1:
            '''
            仅使用源域进行训练
            '''
            #第一次训练开始
            #开始生成目标域模型,并装载源域参数
            name = 'OnlySource'
            target2 = model_define.TargetCNN()    
            
            learning_rate = 0.01
            regularization = 1e-5
            num_epochs = 100
            optimizer = torch.optim.Adam(target2.parameters(), lr = learning_rate , weight_decay = regularization)
            criterion = torch.nn.MSELoss()
            
            for epoch in range(num_epochs):
                prediction = target2.forward(s_xtrain)[1]
                Loss = criterion(prediction , s_ytrain)
                optimizer.zero_grad()
                Loss.backward()
                optimizer.step()  
                if (epoch%20==0):
                    print(epoch,Loss.data)                
            
            #训练目标域模型，测试            
            print('第',i,'个种子时,仅使用源域的效果：')
            nt_Xtest, nt_Ytest_pre = test_TargetCNN(target2, t_xtest)
            result2 = Loss_evalute.loss_evaluate(t_ytest.data.numpy(),nt_Ytest_pre)
            result = np.hstack((result1,result2))
            
            nt_Xtrain, nt_Ytrain_pre = test_TargetCNN(target2, t_xtrain)
            
 
                       
####55555555555555555555555555555555555555555555555555555555全局微调55555555555555555555555555555555555555555555555555555555##### 
        if 1:
            '''
            全局进行微调
            '''
            name = 'FinetuneAll'
            target5 = model_define.TargetCNN()    
            load_TargetCNN(target5,name)               
            learning_rate = 0.0001
            regularization = 1e-4
            epoch = 60
            train_TargetCNN(target5, t_xtrain, t_ytrain, epoch, learning_rate, regularization)
            
            fa_Xtest, fa_Ytest_pre = test_TargetCNN(target5, t_xtest)
            fa_Xtrain, fa_Ytrain_pre = test_TargetCNN(target5, t_xtrain)
            print('第',i,'个种子时,全局微调的效果：')
            result5 = Loss_evalute.loss_evaluate(t_ytest.data.numpy(),fa_Ytest_pre)
            result = np.hstack((result,result5))

#####666666666666666666666666666666666666666666666666666666使用条件核和MSE666666666666666666666666666666666666666666666666666#######         
        if 1:
            '''
            使用CDK条件分布
            '''
            name = 'CDK'
            target6 = model_define.TargetCNN()    
            load_TargetCNN(target6,name)               
            #gamma是MSE的系数，lamda是条件核的系数           
            epoch = 70
            learning_rate = 0.0001
            regularization = 1e-3
            gamma = 0
            lamda = 5
            trainagain_TargetCNN(name,target6, t_xtrain, t_ytrain, t_xtest, epoch, learning_rate, regularization, gamma, lamda) 
            
            cd_Xtest, cd_Ytest_pre = test_TargetCNN(target6, t_xtest)
            cd_Xtrain, cd_Ytrain_pre = test_TargetCNN(target6, t_xtrain)
            print('第',i,'个种子时,使用CDK迁移后的效果：')
            result6 = Loss_evalute.loss_evaluate(t_ytest.data.numpy(),cd_Ytest_pre)
            result = np.hstack((result,result6))
            SeedResult[i,:] = result
            
    name = ['MAE','MAPE','MSE','R2','MAE','MAPE','MSE','R2','MAE','MAPE','MSE','R2','MAE','MAPE','MSE','R2']
    principle = pd.DataFrame(columns=name , data = SeedResult)
    principle.to_csv('seedresult.csv')

    record = pd.DataFrame(data = SeedRecord)
    record.to_csv('seedrecord.csv')
          
          
    