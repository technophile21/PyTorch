
# coding: utf-8

# In[3]:


from __future__ import print_function


# In[4]:


import torch as tc
import torchvision as tcv
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from torchvision import transforms

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import unet.transforms as unet_transforms
from unet.dataset import VOC2012Dataset
from unet.network import UNet

# In[21]:


data_root_path = '../../Datasets/VOC2012/sample'
#data_root_path = '../../Datasets/VOC2012'
        
color_transform = unet_transforms.Colorize()
image_transforms = transforms.Compose([transforms.Scale((572,572)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([.485, .456, .406], [.229, .224, .225])
                                      ])
label_transforms = transforms.Compose([transforms.Scale(572),
                                       unet_transforms.ToLabel()
                                      ])

dataset = VOC2012Dataset(data_root_path, image_transforms, label_transforms)
dataloader = DataLoader(dataset, batch_size=1, num_workers=1)


# In[8]:


def showimage(images):
    img = images.numpy()
    plt.imshow(np.transpose(img, [1,2,0]))
    plt.show()


# In[9]:


#check some data
'''
images, labels = next(iter(dataloader))
showimage(tcv.utils.make_grid(images))
showimage(tcv.utils.make_grid(labels))
'''

# In[10]:


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__()

        self.loss = nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs), targets)
    


# In[ ]:





# In[11]:


def train(args, model):
    model.train()
    
    weight = tc.ones(2)
    weight[0] = 0
    
    optimizer = optim.SGD(params=net.parameters(), lr=0.01, momentum=0.99)
    criterion = CrossEntropyLoss2d(weight)
    
    for epoch in range(1, args['num_epochs'] + 1):
        epoch_loss = []
        
        for step, (images, labels) in enumerate(dataloader):
            inputs = Variable(images)
            targets = Variable(labels)
            outputs = model(inputs)
            
            optimizer.zero_grad()
            
            loss = criterion(outputs, targets[:, 0])
            loss.backward()
            optimizer.step()
            
            epoch_loss.append(loss.data[0])
            
            if args['steps_loss'] > 0 and step % args['steps_loss'] == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print('loss: {average} (epoch: {epoch}, step: {step})')
            if args['steps_save'] > 0 and step % args['steps_save'] == 0:
                filename = '{args.model}-{epoch:03}-{step:04}.pth'
                tc.save(model.state_dict(), filename)
                print('save: {filename} (epoch: {epoch}, step: {step})')
            


# In[12]:


def evaluate(args, model):
    model.eval()
    
    image = input_transform(Image.open(args.image))
    label = model(Variable(image, volatile=True).unsqueeze(0))
    label = color_transform(label[0].data.max[0][1])
    
    ToPILImage(label).save(args.label)
    


# In[ ]:


net = UNet((3, 572, 572))
args = dict(steps_loss=1, steps_save=500, num_epochs=2)
train(args, net)


# In[ ]:




