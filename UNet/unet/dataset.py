
# coding: utf-8


from __future__ import print_function

from torch.utils.data import Dataset as Dataset

import os

from PIL import Image

EXTENSIONS = ['jpg', 'png']

def img_basename(img_path):
    return os.path.basename(os.path.splitext(img_path)[0])
    
def is_image(img_path):
    return any(img_path.endswith(e) for e in EXTENSIONS)
    
class VOC2012Dataset(Dataset):
    def __init__(self, root_path, image_transform=None, label_transform=None):
        super(VOC2012Dataset, self).__init__()
        
        self.root_path = root_path
        self.images_path = os.path.join(root_path, 'images')
        self.labels_path = os.path.join(root_path, 'labels')
        
        print(self.images_path)
        self.filenames = [img_basename(f) for f in os.listdir(self.images_path) if is_image(f)]
        self.filenames.sort()
        
        #Assuming all are images
        self.imageFileNames = os.listdir(self.images_path)
        self.imageFileNames.sort()
        
        self.labelFileNames = os.listdir(self.labels_path)
        self.labelFileNames.sort()
        
        self.image_transform = image_transform
        self.label_transform = label_transform
    
    def __getitem__(self, index):
        
        image_name = self.filenames[index]
        image = Image.open(os.path.join(self.images_path, image_name + '.jpg'), 'r')
        image = image.convert('RGB')
        
        label = Image.open(os.path.join(self.labels_path, image_name + '.png'), 'r')
        label = label.convert('P')
        
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.label_transform is not None:
            label = self.label_transform(label)
            
        return image, label
    
    def __len__(self):
        return len(self.imageFileNames)
    
