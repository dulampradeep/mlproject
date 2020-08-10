#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as tm
import torchvision.datasets as datasets
from tqdm.notebook import tqdm
import warnings

warnings.filterwarnings('ignore')
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[2]:


from customDataset import PlantDataset
import os
import cv2


# In[3]:


os.getcwd()


# In[4]:


os.chdir('F:\\datasets\\leafsnap-dataset\\field')

datadir='F:\\datasets\\leafsnap-dataset'
categories=os.listdir()[:60]
os.chdir(datadir)
os.listdir()


# In[5]:


len(categories)


# In[6]:


labels_list=[]
i=0
for c in categories:
  path=os.getcwd()+"/field/"+c
  class_num=categories.index(c)
  
  for im in os.listdir(path):
    lab=[]
    org=path+"/"+im
    lab.append(org)
    lab.append(class_num)
    labels_list.append(lab)


# In[7]:


len(labels_list)


# In[8]:


test_transforms=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# In[9]:


data1=PlantDataset(labels_list,root_dir=None,transform=test_transforms)
data1


# In[10]:


kkk=len(data1)

l1=int(kkk*0.9)
l2=kkk-l1
print(l1+l2)
print(l1,l2)


# In[11]:


train_data,test_data=torch.utils.data.random_split(data1,[l1,l2])


# In[12]:


print(train_data.__len__())
print(test_data.__len__())


# In[13]:


train_loader=DataLoader(dataset=train_data,batch_size=4,shuffle=True)
test_loader=DataLoader(dataset=test_data,batch_size=4,shuffle=False)


# In[14]:


def show_imgs(imgs,title):
  imgs=imgs.cpu()
  mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
  std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
  imgs = imgs * std + mean
  img_grid=torchvision.utils.make_grid(imgs,nrow=4)
  img_np=img_grid.numpy()
  img_np=np.transpose(img_np,(1,2,0))

  plt.figure(figsize=(8,4))
  plt.imshow(img_np)
  plt.title(title)
  plt.show()


# In[15]:


print("training example")
print(len(train_loader))
i=-1
for batch in train_loader:
  i+=1
  if i==4:
    break
  images,labels=batch
  
  # print(labels)
  show_imgs(images,labels)


# In[17]:


resnet=tm.resnet18(pretrained=True)


# In[18]:


resnet


# In[27]:


resnet.fc=nn.Linear(in_features=resnet.fc.in_features,out_features=60)


# In[140]:


resnet.fc


# In[29]:


for p in resnet.parameters():
  p.requires_grad=False
for p in resnet.fc.parameters():
  p.requires_grad=True
for p in resnet.parameters():
  if(p.requires_grad==True):
    print(p.size())


# In[30]:


def evaluate(model,dataloader):
  total,correct=0,0
  model.eval()
  for batch in dataloader:
    images,labels=batch
    images,labels=images.to(device),labels.to(device)

    outs=model(images)
    out_labels=torch.argmax(outs,axis=1)
    total+=labels.size(0)
    correct+=(labels==out_labels).sum().item()
  return 100*correct/total


# In[31]:


#training with model checkpointing
import copy
def train(model,modelname,loss_fn,optimizer,train_loader,test_loader,epochs):
  hist={'epoch_loss':[],
        'train_acc':[],
        'test_acc':[]}
  min_loss=10000

  for epoch in tqdm(range(1,epochs+1),total=epochs,desc='training last layer'):
    losses=[]
    for batch in train_loader:
      images,labels=batch
      images,labels=images.to(device),labels.to(device)

      model.train()

      outs=model(images)
      loss=loss_fn(outs,labels)
      losses.append(loss.item())

      optimizer.zero_grad()
      loss.backward()

      optimizer.step()

      del images,labels,outs
      torch.cuda.empty_cache()
    
    curr_epoch_loss=np.array(losses).mean()
    hist['epoch_loss'].append(curr_epoch_loss)
    hist['train_acc'].append(evaluate(model,train_loader))
    hist['test_acc'].append(evaluate(model,test_loader))
    
    if curr_epoch_loss<min_loss:
      min_loss=curr_epoch_loss
      best_model=copy.deepcopy(model.state_dict())
  fig,ax=plt.subplots(ncols=2,figsize=(12,6))

  ax[0].plot(range(1,epochs+1),hist['epoch_loss'],label='Loss')
  ax[0].plot(range(1,epochs+1),np.ones(epochs)*min_loss,'r--',alpha=0.6,
             label='Min Loss={}'.format(min_loss))

  ax[0].set_xlabel('Epochs')
  ax[0].set_ylabel('Loss')
  ax[0].set_title("Epochs vs Loss")

  ax[0].grid()
  ax[0].legend()


  ax[1].plot(range(1,epochs+1),hist['train_acc'],'b--',alpha=0.8,label='Train accuracy')

  
  ax[1].plot(range(1,epochs+1),hist['test_acc'],'r--',alpha=0.8,label='Test accuracy')

  ax[1].set_xlabel('Epochs')
  ax[1].set_ylabel('Accuracy')
  ax[1].set_title("Epochs vs score")

  ax[1].grid()
  ax[1].legend()
  plt.plot()
  torch.save(best_model,'{0}_{1:.4f}.pt'.format(modelname,min_loss))
  print('Best loss value : {}'.format(min_loss))

  return best_model


# In[65]:


resnet=resnet.to(device)
opt=torch.optim.Adam(resnet.parameters(),lr=0.0005)
loss_fn=nn.CrossEntropyLoss()
resnet_wts=train(resnet,'resnet',loss_fn,opt,train_loader,test_loader,30)


# In[32]:


p111=torch.load("F:\\datasets\\leafsnap-dataset\\resnet_0.2063.pth")


# In[33]:


model=resnet
model.load_state_dict(p111)
model=model.to(device)
model.eval()
print("Yes")


# In[34]:


test_transforms=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# In[35]:


def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input=image_tensor
    input = input.to(device)
    output = model(input)
    out_labels=torch.argmax(output)
    return out_labels


# In[62]:


import ipywidgets as widgets


# In[126]:


up=widgets.FileUpload(
    accept='image/*',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
    multiple=False  # True to accept multiple files upload else False
)


# In[127]:


up


# In[128]:


[lki]=up.value
lki


# In[129]:


file = open("E:\system\Pictures\ml project\{}".format(lki), "rb")
image = file.read()
widgets.Image(
    value=image,
    format='png',
    width=300,
    height=400,
)


# In[130]:


p=predict_image(cv2.imread("E:\system\Pictures\ml project\{}".format(lki)))
print(categories[p])


# In[131]:


from bs4 import BeautifulSoup as bb
import requests as rr


# In[132]:


res=rr.get("https://en.wikipedia.org/wiki/{}".format(categories[p]))


# In[133]:


bes=bb(res.text,'lxml')


# In[134]:


lin=bes.find('table',class_='infobox biota')


# In[135]:


logo=lin.find("a",class_="image")
tree=logo.img.get('src')


# In[136]:


from IPython.display import Image
from IPython.core.display import HTML 
print("The Image of Tree is :")
Image(url=tree)


# In[137]:


print(lin.find_all('a')[1].text,":",end="")
print(lin.find_all('a')[2].text)
Image(url=lin.find_all('img')[1].get('src'))


# In[138]:


k=lin.find_all("tr")[:15]
# n=[4,5,6,8,10]
# print("Name : ",k[0].text)
print("Details : \n")
t=[]
for i in k:
    j=i.text.split(':')
    t.append(" : ".join([db.replace("\n","") for db in j]))
t[0]="Name : "+t[0]
t[3]+=" : "

for i in t:
    if(i!=""):
        print(i)
        print()
# print(k[3].text,k[4].text.split(":")[1])
# print(k[5].text,k[6].text,k[8].text,k[10].text)

