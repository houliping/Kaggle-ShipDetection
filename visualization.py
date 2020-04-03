import os
import numpy as np
import pandas
import ntpath
import glob
from PIL import Image, ImageDraw
import torch
import torchvision
import settings
import tqdm
import matplotlib.pyplot as plt
config=settings.config

IMAGE_SHAPE = [768, 768]
BAD_IMAGES = ['6384c3e78.jpg']

def rle_decode(mask_rle, shape):
    s=mask_rle.split()
    start,length=[np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    start-=1
    ends=start+length
    img=np.zeros(shape[0]*shape[1],dtype=np.uint8)
    for lo,hi in zip(start,ends):
        img[lo:hi]=1

    img=img.reshape(shape).T
    return img

def mask_to_bbox(mask):
    img_h, img_w = mask.shape[:2]
    rows=np.any(mask,axis=1)
    cols=np.any(mask,axis=0)
    rmin,rmax=np.where(rows)[0][[0,-1]]
    cmin,cmax=np.where(cols)[0][[0,-1]]
    x1 = int(max(cmin - 1, 0))
    y1 = int(max(rmin - 1, 0))
    x2 = int(min(cmax + 1, img_w))
    y2 = int(min(rmax + 1, img_h))

    return x1, y1, x2, y2



def main():
    train_df=pandas.read_csv(config['segmentation'])
    train_df_values = train_df.values
    train_image_to_ann = {}
    print(type(train_df_values))
    image=[]
    mask=[]
    for f in train_df_values:
        image.append(f[0])
        mask.append(f[1])

    print(image[9])
    print(mask[9])

    '''for entry in tqdm.tqdm(train_df_values,"create a table list"):
        image_name=entry[0]
        segmentation=entry[1]
        if image_name in BAD_IMAGES or pandas.isnull(segmentation):
            continue

        print(image_name)
        print(train_df_values.loc)
        mask = rle_decode(segmentation, IMAGE_SHAPE)
        bbox = mask_to_bbox(mask)

        bboxes=np.array(bbox+(0,))
        print(bbox[:4])
        path=os.path.join(config['raw_data_path'],image_name)
        img=Image.open(path)
        draw=ImageDraw.Draw(img)
        draw.rectangle([(bbox[0],bbox[1]),(bbox[2],bbox[3])])
        img.show()'''
        #break






if __name__ == '__main__':
    main()








