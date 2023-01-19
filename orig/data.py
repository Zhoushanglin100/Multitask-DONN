from torchvision import transforms
import torch.utils.data as data
import torch
import glob
import os
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

class DataLoaderSegmentationBin(data.Dataset):
    def __init__(self, folder_path, transform,label=0):
        super(DataLoaderSegmentationBin, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path,'images_prepped_train','*.png'))
        self.mask_files = []
        self.label = label
        self.transform = transform
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(folder_path,'annotations_prepped_train',os.path.basename(img_path)))
        """ 
        self.mapping = {0: 0,
			0.00392157: 1,
			0.00784314: 2,
			0.01176471: 3,
			0.01568628: 4,
			0.01960784: 5,
			0.02352941: 6,
			0.03137255087494850159: 7,
			0.03529412: 8,
			0.03921569: 9,
			0.04313726: 10}
        """
    def mask_to_class_gray(self, mask):
       for k in self.mapping:
           mask[mask == k] = self.mapping[k]
       return mask

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            data = (Image.open(img_path).convert('RGB'))
            if self.transform:
                data = self.transform(data)
            label = np.array(Image.open(mask_path))
            obj_ids = np.unique(label)
            obj_ids = obj_ids[1:]
            labels = (label == obj_ids[:, None, None])
            #print(labels.shape)
            labels = Image.fromarray(labels[self.label])
            if self.transform:
                label = self.transform(labels)
            label_ = torch.where(label!=0,label.double(), -1.0).float()
            return (data).float(), (label_).float()

    def __len__(self):
        return len(self.img_files)





class DataLoaderSegmentation(data.Dataset):
    def __init__(self, folder_path, transform):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path,'images_prepped_train','*.png'))
        self.mask_files = []
        self.transform = transform
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(folder_path,'annotations_prepped_train',os.path.basename(img_path)))
        """
        self.mapping = {0: 0,
			0.00392157: 1,
			0.00784314: 2,
			0.01176471: 3,
			0.01568628: 4,
			0.01960784: 5,
			0.02352941: 6,
			0.03137255087494850159: 7,
			0.03529412: 8,
			0.03921569: 9,
			0.04313726: 10}
        """
    def mask_to_class_gray(self, mask):
       for k in self.mapping:
           mask[mask == k] = self.mapping[k]
       return mask

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            data = (Image.open(img_path).convert('RGB'))
            if self.transform:
                data = self.transform(data)
            label = (Image.open(mask_path))
            #obj_ids = np.unique(label)
            #obj_ids = obj_ids[1:]
            #labels = label == obj_ids[:, None, None]
            #print(labels.shape, labels[0])
            if self.transform:
                label = self.transform(label)
            return (data).float(), (label).float()

    def __len__(self):
        return len(self.img_files)

class load_dataset(DataLoader):
    def __init__(self, batch_size, system_size, datapath, num_workers=8):
        super(DataLoader, self).__init__()
        self.transform = transforms.Compose([transforms.Resize((system_size),interpolation=2),transforms.ToTensor()])
        self.bs = batch_size
        self.datapath = datapath
        self.padding = 0
        self.num_workers = num_workers
    def MNIST(self):
        train_dataset = torchvision.datasets.MNIST(self.datapath, train=True, transform=self.transform, download=True)
        val_dataset = torchvision.datasets.MNIST(self.datapath, train=False, transform=self.transform, download=True)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.bs, num_workers=self.num_workers, shuffle=True, pin_memory=False)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.bs, num_workers=self.num_workers, shuffle=False, pin_memory=False)
        return train_dataloader, val_dataloader
    def MNIST2(self):
        train_dataset = torchvision.datasets.MNIST(self.datapath, train=True, transform=self.transform, download=True)
        val_dataset = torchvision.datasets.MNIST(self.datapath, train=False, transform=self.transform, download=True)
        train_idx = (train_dataset.targets==7)
        val_idx = (val_dataset.targets==7)
        train_dataset.targets = train_dataset.targets[train_idx]
        train_dataset.data = train_dataset.data[train_idx]
        val_dataset.targets = val_dataset.targets[val_idx]
        val_dataset.data = val_dataset.data[val_idx]
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.bs, num_workers=self.num_workers, shuffle=True, pin_memory=False)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.bs, num_workers=self.num_workers, shuffle=False, pin_memory=False)
        return train_dataloader, val_dataloader
    def MNIST3(self,label=7):
        train_dataset = torchvision.datasets.MNIST(self.datapath, train=True, transform=self.transform, download=True)
        val_dataset = torchvision.datasets.MNIST(self.datapath, train=False, transform=self.transform, download=True)
        train_idx = (train_dataset.targets==label)
        val_idx = (val_dataset.targets==label)
        train_dataset.targets = train_dataset.targets[train_idx]
        train_dataset.data = train_dataset.data[train_idx]
        val_dataset.targets = val_dataset.targets[val_idx]
        val_dataset.data = val_dataset.data[val_idx]
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.bs, num_workers=self.num_workers, shuffle=True, pin_memory=False)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.bs, num_workers=self.num_workers, shuffle=False, pin_memory=False)
        return train_dataloader, val_dataloader
    def MNIST4(self,label=[7,2]):
        train_dataset = torchvision.datasets.MNIST(self.datapath, train=True, transform=self.transform, download=True)
        val_dataset = torchvision.datasets.MNIST(self.datapath, train=False, transform=self.transform, download=True)
        train_idx = train_dataset.targets==label[0]
        val_idx = val_dataset.targets==label[0]
        for i in range(len(label)):
            if i == 0:
                continue
            train_idx = torch.logical_or( train_idx , (train_dataset.targets==label[i]))
            val_idx = torch.logical_or( val_idx, (val_dataset.targets==label[i]))
        train_dataset.targets = train_dataset.targets[train_idx]
        train_dataset.data = train_dataset.data[train_idx]
        val_dataset.targets = val_dataset.targets[val_idx]
        val_dataset.data = val_dataset.data[val_idx]
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.bs, num_workers=self.num_workers, shuffle=True, pin_memory=False)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.bs, num_workers=self.num_workers, shuffle=False, pin_memory=False)
        return train_dataloader, val_dataloader



    def FMNIST(self):
        train_dataset = torchvision.datasets.FashionMNIST(self.datapath, train=True, transform=self.transform, download=True)
        val_dataset = torchvision.datasets.FashionMNIST(self.datapath, train=False, transform=self.transform, download=True)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.bs, num_workers=self.num_workers, shuffle=True, pin_memory=False)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.bs, num_workers=self.num_workers, shuffle=False, pin_memory=False)
        return train_dataloader, val_dataloader


    def FMNIST4(self,label=[7,2]):
        train_dataset = torchvision.datasets.FashionMNIST(self.datapath, train=True, transform=self.transform, download=True)
        val_dataset = torchvision.datasets.FashionMNIST(self.datapath, train=False, transform=self.transform, download=True)
        train_idx = train_dataset.targets==label[0]
        val_idx = val_dataset.targets==label[0]
        for i in range(len(label)):
            if i == 0:
                continue
            train_idx = torch.logical_or( train_idx , (train_dataset.targets==label[i]))
            val_idx = torch.logical_or( val_idx, (val_dataset.targets==label[i]))
        train_dataset.targets = train_dataset.targets[train_idx]
        train_dataset.data = train_dataset.data[train_idx]
        val_dataset.targets = val_dataset.targets[val_idx]
        val_dataset.data = val_dataset.data[val_idx]
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.bs, num_workers=self.num_workers, shuffle=True, pin_memory=False)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.bs, num_workers=self.num_workers, shuffle=False, pin_memory=False)
        return train_dataloader, val_dataloader



