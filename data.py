from torch.utils.data import Dataset
import numpy as np
import glob
import ntpath
import os
import math
import random
import torch
import shutil
import settings
from torchvision import transforms
from PIL import Image
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
config=settings.config


def cut_empty(names,segmentation_df:pandas.DataFrame):
        return [name for name in names
                if(type(segmentation_df.loc[ntpath.basename(name)]['EncodedPixels'])!=float)]

def split_for_val():
    '''
    split for validation with the ratio of 8:2
    :param dataPath:dataRoot
    :return:
    '''

    img_index=glob.glob(config['raw_data_path']+"/*.*")#define the listdir
    for img in img_index:
        if ntpath.basename(img) in config['BAD_IMAGES']:
            print("processing %s"%img)
            img_index.remove(img)
    train_n,val_n=train_test_split(img_index,test_size=0.05,random_state=42)
    segmentation_df=pandas.read_csv(config['segmentation']).set_index("ImageId")
    train_n=cut_empty(train_n,segmentation_df)
    val_n=cut_empty(val_n,segmentation_df)#remove the data which is not included a ship

    return train_n,val_n,segmentation_df


class DataSet2(Dataset):
    def __init__(self,phase="train",train_n=None,val_n=None,segmentation_df=None,test_path=None,):
        assert phase=='train' or phase=='val' or phase=='test'
        self.phase=phase
        self.segmentation_df=segmentation_df
        self.train_n=train_n
        self.val_n=val_n
        assert phase=="train" or phase=="val" or phase=="test"
        self.image_to_tensor=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
        if self.phase=='test':
            self.test_n=glob.glob(test_path+"/*.*")



    def __len__(self):
        if self.phase=='train':
            return int(len(self.train_n))
        elif self.phase=='val':
            return int(len(self.val_n))
        else:
            return int(len(self.test_n))
    def __getitem__(self, item):
        if self.phase=='train': train_val=self.train_n
        elif self.phase=='val': train_val=self.val_n
        else: train_val=self.test_n

        if item>len(train_val):
            item=int(item%train_val)

        image_pah=train_val[item]
        img=Image.open(image_pah)
        img_tensor=self.image_to_tensor(img)
        img_name=ntpath.basename(image_pah)
        if self.phase=='train' or self.phase=='val':
            mask_lable=self.get_mask(img_name,self.segmentation_df)
            return img_tensor,torch.from_numpy(mask_lable),img_name
        else:
            return img_tensor,img_name

    def get_mask(self,img_id, df):
        shape = (768,768)
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        masks = df.loc[img_id]['EncodedPixels']
        if(type(masks) == float): return img.reshape(shape)
        if(type(masks) == str): masks = [masks]
        for mask in masks:
            s = mask.split()
            for i in range(len(s)//2):
                start = int(s[2*i]) - 1
                length = int(s[2*i+1])
                img[start:start+length] = 1
        return img.reshape(shape).T

    def rle_decode(self,mask_rle, shape):
        img=np.zeros(shape[0]*shape[1],dtype=np.uint8)
        print(type(mask_rle))
        s=mask_rle.split()
        start,length=[np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        start-=1
        ends=start+length

        for lo,hi in zip(start,ends):
            img[lo:hi]=1

        img=img.reshape(shape).T
        return img




class DataSet(Dataset):
    def __init__(self,phase="train",val_path=None):
        assert phase=="train" or phase=="val" or phase=="test"
        self.image_to_tensor=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        )
        self.img_index=glob.glob(config['raw_data_path']+"/*.*")
        self.train_segmentation=pandas.read_csv(config['segmentation']).values
        self.name=[]
        self.mask=[]
        self.bbox=[]
        self.pos_number=0
        self.neg_number=0
        for f in self.train_segmentation:
            self.name.append(f[0])
            self.mask.append(f[1])
            if pandas.isnull(f[1]):
                self.neg_number+=1
            else:
                self.pos_number+=1
        print("the neg is %d"%self.neg_number)
        print("the pos is %d"%self.pos_number)


    def __len__(self):
        return int(len(self.img_index))

    def __getitem__(self, item):
        if item>len(self.img_index):
            item=int(item%self.img_index)
        image_pah=self.img_index[item]
        img=Image.open(image_pah)
        img_tensor=self.image_to_tensor(img)
        img_name=ntpath.basename(image_pah)
        index=self.name.index(img_name)
        mask_rle=self.mask[index]
        mask_lable=self.rle_decode(mask_rle,settings.config['shape'])

        return img_tensor,mask_lable


    def rle_decode(self,mask_rle, shape):
        img=np.zeros(shape[0]*shape[1],dtype=np.uint8)
        if pandas.isnull(mask_rle):
            img=img.reshape(shape).T
            return img

        s=mask_rle.split()
        start,length=[np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        start-=1
        ends=start+length

        for lo,hi in zip(start,ends):
            img[lo:hi]=1

        img=img.reshape(shape).T
        return img


if __name__ == '__main__':
    train_n,val_n,segmentation_df=split_for_val()
    dataset=DataSet2(phase="train",train_n=train_n,segmentation_df=segmentation_df)
    dataset[2]