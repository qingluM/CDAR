# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 20:32:14 2020

@author: Qinglu Meng
"""
import numpy as np
import torch
import Loss_evalute
import conditional_kernel
import pandas as pd
import model_define
import scipy.io as sio


def load_TargetCNN(model,name,Task):

    if (Task=='5-7'):    
        pretrained_dict = torch.load("pre-trained model/model used in paper/"+Task+".pth")
               
    elif (Task=='6-7'):        
        pretrained_dict = torch.load("pre-trained model/model used in paper/"+Task+".pth")
               
    elif (Task=='7-5'):        
        pretrained_dict = torch.load("pre-trained model/model used in paper/"+Task+".pth")
              
    elif (Task=='7-6'):        
        pretrained_dict = torch.load("pre-trained model/model used in paper/"+Task+".pth")
     
    new_dict = model.state_dict()
    pretrained_dict1 = {k: v for k, v in pretrained_dict.items() if k in new_dict}
    new_dict.update(pretrained_dict1)
    model.load_state_dict(new_dict)
    
    if name=='CDAR':
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
   
    print('Moldel loading finished')   
        

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

def trainagain_TargetCNN(name, model, t_xtrain, t_ytrain, s_xtrain, epoch, learning_rate, regularization, Lambda, Beta):
    
    params = filter(lambda p: p.requires_grad, model.parameters())
    
    lr = learning_rate
    fc1_params = list(map(id, model.fc1.parameters()))
    fc2_params = list(map(id, model.fc2.parameters()))
    fc3_params = list(map(id, model.fc3.parameters()))

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
            
        CEOD_loss = conditional_kernel.MLcon_kernel(source_list, prediction, target_list, t_ytrain)
        Loss = Lambda*criterion(prediction2, t_ytrain) + Beta*CEOD_loss            
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
    
    '''
    Select the corresponding task
    '''
    Task = '5-7'
    #Task = '6-7'
    #Task = '7-5'
    #Task = '7-6'
                 
    #Import data  
    battery7 = pd.read_csv("data/7.csv", header=None).values.reshape(168,1,371)
    label7 = pd.read_csv("data/L7.csv", header = None).values
    
    battery6 = pd.read_csv("data/6.csv", header=None).values.reshape(168,1,371)
    label6 = pd.read_csv("data/L6.csv", header = None).values
    
    battery5 = pd.read_csv("data/5.csv", header=None).values.reshape(168,1,371)
    label5 = pd.read_csv("data/L5.csv", header = None).values
    
    if (Task=='5-7'):
        
        seedrecord = pd.read_csv("SeedRecord/Seed-"+Task+".csv").values       
        source_x = torch.Tensor(battery5)
        source_y = torch.Tensor(label5)
        target_x = torch.Tensor(battery7)
        target_y = torch.Tensor(label7)
        print(Task)
        
    elif (Task=='6-7'):
        
        seedrecord = pd.read_csv("SeedRecord/Seed-"+Task+".csv").values
        source_x = torch.Tensor(battery6)
        source_y = torch.Tensor(label6)
        target_x = torch.Tensor(battery7)
        target_y = torch.Tensor(label7)   
        print(Task)
        
    elif (Task=='7-5'):
        
        seedrecord = pd.read_csv("SeedRecord/Seed-"+Task+".csv").values
        source_x = torch.Tensor(battery7)
        source_y = torch.Tensor(label7)
        target_x = torch.Tensor(battery5)
        target_y = torch.Tensor(label5)  
        print(Task)
             
    elif (Task=='7-6'):
        
        seedrecord = pd.read_csv("SeedRecord/Seed-"+Task+".csv").values
        source_x = torch.Tensor(battery7)
        source_y = torch.Tensor(label7)
        target_x = torch.Tensor(battery6)
        target_y = torch.Tensor(label6) 
        print(Task)
        
        
    SeedResult = np.zeros((25,16))
    
    for i in range(25):
                
        index1 = seedrecord[i,:]
        index2 = np.delete(np.arange(168),index1)
               
        t_xtrain = target_x[index1,:,:]
        t_ytrain = target_y[index1,:]
        t_xtest = target_x[index2,:,:]
        t_ytest = target_y[index2,:]
        
        s_xtrain = source_x
        s_ytrain = source_y
    

##############################################    
        if 1: #Model 1
        
            name = 'OnlyTarget'
            target1 = model_define.TargetCNN()
            learning_rate = 0.03
            regularization  = 1e-3
            num_epochs = 100
            
            optimizer = torch.optim.Adam(target1.parameters(), lr = learning_rate , weight_decay = regularization)
            criterion = torch.nn.MSELoss()           
            print('Start')
            
            for epoch in range(num_epochs):
                prediction = target1.forward(t_xtrain)[1]
                Loss = criterion(prediction , t_ytrain)
                optimizer.zero_grad()
                Loss.backward()
                optimizer.step()  
                if (epoch%20==0):
                    print(epoch,Loss.data)
                    

            ot_Xtest, ot_Ytest_pre = test_TargetCNN(target1, t_xtest)
            print('Results of No',i,'seed(Model 1):')
            result1 = Loss_evalute.loss_evaluate(t_ytest.data.numpy(),ot_Ytest_pre)
            print('---------------------------')     
            

##############################################
        if 1: #Model 2

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
                    
            nt_Xtest, nt_Ytest_pre = test_TargetCNN(target2, t_xtest)        
            print('Results of No',i,'seed(Model 2):')                     
            result2 = Loss_evalute.loss_evaluate(t_ytest.data.numpy(),nt_Ytest_pre)
            print('---------------------------')
            result = np.hstack((result1,result2))
            
################################################
        if 1: # method FA

            name = 'FinetuneAll'
            target3 = model_define.TargetCNN()    
            load_TargetCNN(target3,name,Task)               
            learning_rate = 0.001
            regularization = 1e-4
            epoch = 60
            train_TargetCNN(target3, t_xtrain, t_ytrain, epoch, learning_rate, regularization)
            
            fa_Xtest, fa_Ytest_pre = test_TargetCNN(target3, t_xtest)
            fa_Xtrain, fa_Ytrain_pre = test_TargetCNN(target3, t_xtrain)
            print('Results of No',i,'seed(FA):')
            result3 = Loss_evalute.loss_evaluate(t_ytest.data.numpy(),fa_Ytest_pre)
            print('---------------------------')
            result = np.hstack((result,result3))

################################################         
        if 1: # mehod CDAR
 
            name = 'CDAR'
            target4 = model_define.TargetCNN()    
            load_TargetCNN(target4,name,Task)
               
            # Lambda*MSE + Beta*CEOD
            epoch = 0
            learning_rate = 0.001
            regularization = 1e-4
            Lambda = 0
            Beta = 0
            trainagain_TargetCNN(name,target4, t_xtrain, t_ytrain, t_xtest, epoch, learning_rate, regularization, Lambda, Beta) 
            
            cd_Xtest, cd_Ytest_pre = test_TargetCNN(target4, t_xtest)
            cd_Xtrain, cd_Ytrain_pre = test_TargetCNN(target4, t_xtrain)
            print('Results of No',i,'seed(CDAR):')
            result4 = Loss_evalute.loss_evaluate(t_ytest.data.numpy(),cd_Ytest_pre)
            print('---------------------------')
            result = np.hstack((result,result4))
      
            SeedResult[i,:] = result
            
            i_str = str(i)
            filename = 'Process results/'+Task+'-seed-'+ i_str+'.mat'
            sio.savemat(filename, {'seed':index1,
                                   'ot_Ytest_pre':ot_Ytest_pre,'ot_Xtest':ot_Xtest[2],
                                   'nt_Ytest_pre':nt_Ytest_pre,'nt_Xtest':nt_Xtest[2],
                                   'fa_Ytest_pre':fa_Ytest_pre,'fa_Xtest':fa_Xtest[2],
                                   'fa_Ytrain_pre':fa_Ytrain_pre,'fa_Xtrain':fa_Xtrain[2],
                                   'cd_Ytest_pre':cd_Ytest_pre,'cd_Xtest':cd_Xtest[2],
                                   'cd_Ytrain_pre':cd_Ytrain_pre,'cd_Xtrain':cd_Xtrain[2],                              
                                   'Ytest':t_ytest.data.numpy(),
                                   'Ytrain':t_ytrain.data.numpy()})             


    name = ['MAE','MSE','MAPE','R2','MAE','MSE','MAPE','R2','MAE','MSE','MAPE','R2','MAE','MSE','MAPE','R2']
    principle = pd.DataFrame(columns=name , data = SeedResult)
    principle.to_csv('Results/'+Task+'seedresult.csv')
          
          
    