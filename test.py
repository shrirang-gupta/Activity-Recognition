import pandas as pd 
import ast    #evaluating python expressions kind of stuff
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split
import torchvision as tv
from torchvision import transforms

class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)) 
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.maxpool2=nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0, dilation=1, ceil_mode=False)
        # self.flatten=nn
        self.fc1=nn.Linear(32768, 12)

    def forward(self, x):
        x = self.conv1(x) 
        x = self.maxpool1(x) # x = self.maxpool1(F.relu(x))
        x = self.conv2(F.relu(x))
        x = self.maxpool2(F.relu(x))
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        x = self.fc1(F.relu(x))  #should we apply two or more linear function
        # print("x", x.shape)
        x=F.softmax(x, dim=1)
        return x

class DatasetLoader(Dataset):
    def __init__(self,datax,datay):
        self.datax=datax 
        self.datay=datay
        

    def __len__(self):
        return len(self.datay)

    def __getitem__(self, index):

        image=self.datax[index]
    
        # print("index",index)
        # print("self.datay.loc[index",self.datay.loc[index,self.datay.columns[1:]])
        labels=self.datay.reset_index().loc[index,self.datay.columns[1:]] 
        return torch.Tensor(image),torch.Tensor(labels)

EPOCHS = 10
def train_model(train_data,model,optimizer,Loss):
    for epoch in range(EPOCHS):
        for i, data in enumerate(train_data):
            inputs, labels = data
            # print("inputs", inputs.shape)
            inputs = torch.reshape(inputs, (inputs.shape[0], inputs.shape[4], inputs.shape[2], inputs.shape[3], inputs.shape[1]))
            pred=model(inputs)
            # print("pred", pred.shape)
            loss=Loss(pred,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print("------------epoch---------",epoch,"-----loss------",loss.item())


def val_model(test_data,model,Loss):
    v_e_loss=0
    prediction=[]
    labels_e=[]
    total_y=0
    correct_y=0
    model.eval() #turns off certain layers like Drop out layers during evaluation
    with torch.no_grad():
        for i, data in enumerate(test_data):
            inputs, labels = data
            inputs = torch.reshape(inputs, (inputs.shape[0], inputs.shape[4], inputs.shape[2], inputs.shape[3], inputs.shape[1]))

            pred=model(inputs)
            loss=Loss(pred,labels)
            v_e_loss+=loss.item()
            outputs=torch.sigmoid(pred)
            pred=torch.round(outputs)
            prediction.append(pred)
            labels_e.append(labels)
            predicted=pred
            total_y+=labels.size(0)
            correct_y+=predicted.eq(labels).sum().item()
        v_loss=round(v_e_loss/len(test_data),2)
        v_accuracy=round(100*correct_y/total_y,2)
        
        print('Validation accuracy: {}%'.format(v_accuracy), (correct_y, '/',total_y))
    

if __name__ == "__main__":
    x_labels = pd.read_csv(r"E:\content\Hollywood2_final\labels\train3.csv")
    with open('video_array_small.npy', 'rb') as f:
        x_array = np.load(f)
    xtrain,xval,ytrain,yval=train_test_split(x_array,x_labels,test_size=0.2,random_state=42)
    print("train shape",xtrain.shape)
    print("val shape",xval.shape)

    #create dataloader
    trainloader = DatasetLoader(xtrain,ytrain)
    valloader = DatasetLoader(xval,yval)

    #load data
    train_data = DataLoader(trainloader, batch_size=32, shuffle=True)
    val_data = DataLoader(valloader, batch_size=32, shuffle=True)
    res=model()
    optimizer=optim.Adam(res.parameters(), lr=0.001, weight_decay=0.001)
    Loss=nn.BCEWithLogitsLoss(reduction="mean") #using for multilabel

    train_model(train_data,res,optimizer,Loss)
    val_model(val_data,res,Loss)



