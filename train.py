import os
import ntpath
import torch
import sys
import numpy as np
import settings
import argparse
from importlib import import_module
from data import *
from torch.utils.data import DataLoader
from torch.nn import BCELoss,DataParallel
import warnings
import time
from torch.autograd import Variable
import torch.nn.functional as F
from visual_loss import *
import Network.unet_model as unet_model
import Network.ResNet50Unet as ResNet50Unet
from torch.backends import cudnn
from Unet34.loss import MixedLoss
from Unet34.metric import IOU,dice,mean_fscore
config=settings.config
from PIL import Image
import scipy.misc

def argv_args():
    parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
    parser.add_argument('--model', '-m', metavar='MODEL', default='base',
                    help='model')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--save-freq', default=2, type=int, metavar='S',
                        help='save frequency')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--save-dir', default='./resnet50/', type=str, metavar='SAVE',
                        help='directory to save checkpoint (default: none)')
    parser.add_argument('--test', default=0, type=int, metavar='TEST',
                        help='1 do test evaluation, 0 not')
    parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                        help='In the test phase, split the image to 8 parts')
    parser.add_argument('--gpu', default='all', type=str, metavar='N',
                        help='use gpu')
    parser.add_argument('--n_test', default=8, type=int, metavar='N',
                        help='number of gpu for test')
    parser.add_argument('--visdom',default='Match',type=str,metavar='V',help='visdom window')

    args=parser.parse_args()
    return args

def main():
    global args
    args=argv_args()
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    #net=import_module(args.model).get_model(config['channels'],config['classes'])
    net=ResNet50Unet.get_model(config['channels'],config['classes'])
    net=net.cuda()

    train_n,val_n,segmentation_df=split_for_val()
    cudnn.benchmark = True
    dataset=DataSet2(phase="train",train_n=train_n,segmentation_df=segmentation_df)
    dataset_val=DataSet2(phase="val",val_n=val_n,segmentation_df=segmentation_df)
    dataset=DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.workers,
        shuffle=True
    )
    dataset_val_loader=DataLoader(
        dataset=dataset_val,
        batch_size=1,
        pin_memory=True,
        num_workers=args.workers,
        shuffle=True
    )
    optimizer=torch.optim.SGD(
        net.parameters(),
        args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    def get_lr(epoch):
        if epoch <= args.epochs * 0.5:
            lr = args.lr
        elif epoch <= args.epochs * 0.8:
            lr = 0.1 * args.lr
        else:
            lr = 0.01 * args.lr
        return lr
    startepochs=args.start_epoch+1
    print("start training......")
    visual=Visual_loss(args.visdom)
    loss=MixedLoss(alpha=settings.hyper_parameter['alpha'],gamma=settings.hyper_parameter['gamma'])
    net=DataParallel(net,device_ids=[0,1,2,3])
    #visual=Visual_loss(args.visdom)

    #---------for validation--------------
    '''if True:
        load_model=torch.load("10.ckpt")
        net.load_state_dict(load_model['state_dict'])
        net.cuda()
        validation(data_loader=dataset_val,net=net,loss=loss)

        return'''
    #-------------------------------------



    for epoch in range(startepochs, args.epochs + 1):
        train(dataset,net,loss,epoch,optimizer,get_lr,args.save_dir)
        validation(data_loader=dataset_val_loader,net=net,loss=loss,epoch=epoch)

static_count=0#对其进行计数
def train(data_loader, net,loss,epoch, optimizer, get_lr, save_dir,visual=None):
    global static_count
    warnings.filterwarnings('ignore')
    start_time = time.time()
    net.train()
    lr=get_lr(epoch)
    for para_group in optimizer.param_groups:
        para_group['lr']=lr

    for i,(img,mask,img_name) in enumerate(data_loader):
        img=Variable(img.cuda(async=True))
        mask=Variable(mask.cuda())

        out_pred=net(img)
        #masks_probs = F.sigmoid(out_pred)
        #masks_probs_flat=masks_probs.view(-1)
        #mask_label_flat=mask.view(-1)
        loss_data=loss(out_pred,mask.float())
        #visual.show_loss_curve(total_loss=loss_data.item(),X=static_count)
        dice_score=dice(out_pred,mask.float())
        mean_fscore_=mean_fscore(out_pred,mask.float())
        #visual.show_dice(dice_score.item(),static_count,'dice_score')
        print("--------------------------------------------------")
        print("the loss_data is %f"%loss_data.item())
        print("the dice score is %f"%dice_score.item())
        print("the fs score is %f"%mean_fscore_.item())
        print("--------------------------------------------------")
        static_count+=1
        optimizer.zero_grad()
        loss_data.backward()
        optimizer.step()
    if epoch%args.save_freq==0:
        state_dict=net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key]=state_dict[key].cpu()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save({
            'epoch':epoch,
            'state_dict':state_dict,
            'save_dir':save_dir
        },os.path.join(save_dir,'%3d.ckpt'%epoch))


def validation(data_loader,net,loss,epoch,visual=None):
    warnings.filterwarnings('ignore')
    start_time = time.time()
    net.eval()
    if not os.path.exists("./pred_mask/"):
        os.makedirs("./pred_mask/")
    for i,(img,mask,img_name) in enumerate(data_loader):
        img=img.cuda(async=True)
        mask=mask.cuda()
        out_pred=net(img)
        loss_data=loss(out_pred,mask.float())
        dice_score=dice(out_pred,mask.float())
        mean_fscore_=mean_fscore(out_pred,mask.float())
        print("----------------------The Epoch %d-----------------------"%epoch)
        print("the loss_data is %f"%loss_data.item())
        print("the dice score is %f"%dice_score.item())
        print("the fs score is %f"%mean_fscore_.item())
        print("---------------------------------------------------------")
        #visual.show_dice_val(dice_score,epoch,"dice_val")

        '''iou=IOU(out_pred,mask.float())
        print("the IOU is: ",iou.item())
        out_pred=(out_pred>0).float()
        out_pred=out_pred.squeeze(0).squeeze(0)
        out_pred=out_pred.contiguous().cpu().detach().numpy()
        print(out_pred.shape)
        scipy.misc.imsave(os.path.join("./pred_mask",img_name),out_pred)
        #im=Image.fromarray(out_pred)
        #im.save(os.path.join("./pred_mask",img_name))
        #out_pred=out_pred.cpu().numpy()'''







if __name__ == '__main__':
    main()



