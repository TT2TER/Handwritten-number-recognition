import copy
import time
import pandas as pd
import torch
import torch.utils.data as Data
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transforms


train_data = dataset.MNIST(root='./data/MNIST',train=True,transform=transforms.ToTensor(),download=True)
test_data = dataset.MNIST(root='./data/MNIST',train=False,transform=transforms.ToTensor(),download=True)

train_loader = Data.DataLoader(train_data,batch_size=64,shuffle=True,num_workers=0)
test_loader = Data.DataLoader(test_data,batch_size=10000,shuffle=False,num_workers=0)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120,84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84,10)
        self.soft= nn.Softmax(dim=1)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        #
        x = x.view(x.size()[0],-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.soft(x)
        return x

EPOCH = 12
BATCH_SIZE = 64
LR = 0.01

def train_lenet(model,train_loader,test_loader,criterion,optimizer,num_epochs = EPOCH):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    train_loss_all = []
    train_acc_all = []
    test_loss_all = []
    test_acc_all = []
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs-1))
        print('-'*10)
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        test_loss = 0.0
        test_corrects = 0
        test_num = 0
        for step,(batch_img,batch_label) in enumerate(train_loader):
            model.train()
            output = model(batch_img)
            pre_lab = torch.argmax(output,1)
            loss = criterion(output,batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*batch_img.size(0)
            train_corrects += torch.sum(pre_lab==batch_label.data)
            train_num += batch_img.size(0)
        train_loss_all.append(train_loss/train_num)
        train_acc_all.append(train_corrects.double().item()/train_num)
        for step, (batch_img,batch_label)in enumerate(test_loader):
            model.eval()
            output = model(batch_img)
            pre_lab = torch.argmax(output,1)
            loss = criterion(output,batch_label)
            test_loss += loss.item()*batch_img.size(0)
            test_corrects += torch.sum(pre_lab==batch_label.data)
            test_num +=batch_img.size(0)
        test_loss_all.append(test_loss/test_num)
        test_acc_all.append(test_corrects.double().item()/test_num)

        print('{} Train Loss: {:.4f}   Train Acc: {:.4f}'.format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print('{} Test Loss: {:.4f}   Test Acc: {:.4f}'.format(epoch,test_loss_all[-1],test_acc_all[-1]))
        save_path = './model/model_{}.pth'.format(epoch)
        torch.save(my_model, save_path)

        if test_acc_all[-1]>best_acc:
            best_acc = test_loss_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        time_use = time.time()-since
        print('Train and val complete in {:.0f}m {:.0f}s'.format(time_use//60,time_use%60))

    model.load_state_dict(best_model_wts)
    train_process = pd.DataFrame(
        data={"epoch":range(num_epochs),
            "train_loss_all":train_loss_all,
            "test_loss_all":test_loss_all,
            "train_acc_all":train_acc_all,
            "test_acc_all":test_acc_all}
    )
    return model,train_process

if __name__== '__main__':
    my_model = LeNet()
    optimizer = torch.optim.SGD(my_model.parameters(), lr=LR, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    my_model,train_process = train_lenet(my_model,train_loader,test_loader,criterion,optimizer)
    train_process.to_csv('./model/loss.csv')








