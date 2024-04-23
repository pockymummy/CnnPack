# =============================================================================
# Code to train EDoF CNNS models
# =============================================================================

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['cervix93','fraunhofer','fraunhofer_separate','fraunhofer_elastic','fraunhofer_elastic_only', 'fraunhofer_raw'], default='fraunhofer_elastic_only')
parser.add_argument('--image_size', choices=[512,640], default=512)
parser.add_argument('--method', choices=[
    'EDOF_CNN_max','EDOF_CNN_3D','EDOF_CNN_concat','EDOF_CNN_backbone','EDOF_CNN_fast','EDOF_CNN_RGB','EDOF_CNN_pairwise','EDOF_CNN_pack', 'PackNet01'], default='EDOF_CNN_pack_43')
parser.add_argument('--Z', choices=[3,5,7,9], type=int, default=5)
parser.add_argument('--fold', type=int, choices=range(5),default=1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--cudan', type=int, default=0)
parser.add_argument('--image_channels', choices=['rgb','grayscale'], default='3channels')
parser.add_argument('--automate', choices=[0,1], default=0)
parser.add_argument('--augmentation', choices=[0,1], default=0)
parser.add_argument('--augmentLevel', choices=[0,1], default=0)
parser.add_argument('--rotate', choices=[0,1,2,3,4], default=0)
parser.add_argument('--hflip', choices=[0,1], default=0)
parser.add_argument('--ssim', choices=[0,1], default=0)
parser.add_argument('--jitter', choices=[0,1], default=2)
parser.add_argument('--comment', choices=['',''], default='rgbjitter')
args = parser.parse_args()

import numpy as np
from time import time
from torch import optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import torch
import dataset, models
from tqdm import tqdm
from PIL import Image
import os
from torchvision import transforms

device = torch.device('cuda:'+str(args.cudan) if torch.cuda.is_available() else 'cpu')

#define transforms if rgb or not
if args.image_channels=='rgb' or args.image_channels=='3channels':
    if args.rotate==1:
        if args.augmentLevel==0:
            train_transform=dataset.aug_transforms_rgb_aug
        elif args.augmentLevel==1:
            train_transform=dataset.aug_transforms_rgb_aug_1
        elif args.augmentLevel==2:
            train_transform=dataset.aug_transforms_rgb_aug_2
    else:
        train_transform=dataset.aug_transforms_rgb
    if args.rotate ==1:
        common_transform = dataset.common_transform_rotate
    else:
        common_transform = dataset.common_transform_empty
    test_transform=dataset.val_transforms_rgb
    train_transformY=dataset.val_transforms_rgb
    test_transformY=dataset.val_transforms_rgb
    test_common_transform=dataset.common_transform_empty
elif args.image_channels=='red':
    if args.comment=='rgbjitter':
        train_transform=dataset.transforms_PIL_preprocess
        test_transform=dataset.transforms_PIL_preprocess
        train_transformY=dataset.transforms_PIL_preprocess
        test_transformY=dataset.transforms_PIL_preprocess
        common_transform=transforms.Compose(dataset.transform_jitter_2 + dataset.transforms_red_pil + dataset.transforms_to_tensor)
        test_common_transform = transforms.Compose(dataset.transforms_red_pil + dataset.transforms_to_tensor)
    else:
        train_transform=dataset.aug_transforms_red
        test_transform=dataset.aug_transforms_red
        train_transformY=dataset.aug_transforms_red
        test_transformY=dataset.aug_transforms_red
        if args.jitter==1:
            common_transform=dataset.common_transform_jitter_1
        else:
            common_transform=dataset.common_transform_empty
elif args.image_channels=='green':
    if args.comment=='rgbjitter':
        train_transform=dataset.transforms_PIL_preprocess
        test_transform=dataset.transforms_PIL_preprocess
        train_transformY=dataset.transforms_PIL_preprocess
        test_transformY=dataset.transforms_PIL_preprocess
        common_transform=transforms.Compose(dataset.transform_jitter_2 + dataset.transforms_green_pil + dataset.transforms_to_tensor)
        test_common_transform = transforms.Compose(dataset.transforms_green_pil + dataset.transforms_to_tensor)
    else:
        train_transform=dataset.aug_transforms_green
        test_transform=dataset.aug_transforms_green
        train_transformY=dataset.aug_transforms_green
        test_transformY=dataset.aug_transforms_green
        common_transform=dataset.common_transform_empty
        if args.jitter==1:
            common_transform=dataset.common_transform_jitter_1
        else:
            common_transform=dataset.common_transform_empty
elif args.image_channels=='blue':
    if args.comment=='rgbjitter':
        train_transform=dataset.transforms_PIL_preprocess
        test_transform=dataset.transforms_PIL_preprocess
        train_transformY=dataset.transforms_PIL_preprocess
        test_transformY=dataset.transforms_PIL_preprocess
        common_transform=transforms.Compose(dataset.transform_jitter_2 + dataset.transforms_blue_pil + dataset.transforms_to_tensor)
        test_common_transform = transforms.Compose(dataset.transforms_blue_pil + dataset.transforms_to_tensor)
    else:
        train_transform=dataset.aug_transforms_blue
        test_transform=dataset.aug_transforms_blue
        train_transformY=dataset.aug_transforms_blue
        test_transformY=dataset.aug_transforms_blue
        common_transform=dataset.common_transform_empty
        if args.jitter==1:
            common_transform=dataset.common_transform_jitter_1
        else:
            common_transform=dataset.common_transform_empty
else:
    if args.augmentation==1:
        if args.augmentLevel==0:
            train_transform=dataset.aug_transforms_aug
        elif args.augmentLevel==1:
            train_transform=dataset.aug_transforms_aug_1
    else:
        train_transform=dataset.aug_transforms
    if args.rotate ==1:
        common_transform = dataset.common_transform_rotate
    elif args.rotate == 2:
        common_transform = dataset.common_transform_rotate_2
    elif args.rotate == 3:
        common_transform = dataset.common_transform_rotate_3
    elif args.rotate == 4:
        common_transform = dataset.common_transform_rotate_4
    elif args.hflip == 1:
        common_transform = dataset.common_transform_hflip
    else:
        common_transform = dataset.common_transform_empty
    test_transform=dataset.val_transforms
    train_transformY=dataset.val_transforms
    test_transformY=dataset.val_transforms

def normalize8(I):
  mn = I.min()
  mx = I.max()

  mx -= mn

  I = ((I - mn)/mx) * 255
  return I.astype(np.uint8)

def view_images(model,tst):
    Yhats=[]
    Ytrues=[]
    stacks=[]
    model.eval()
    with torch.no_grad():
        for XX, Y in tst:
              XX = [X.to(device) for X in XX]
              Y = Y.to(device, torch.float)
              Yhat = model(XX)
              Yhats.append(Yhat[0].cpu().numpy())
              Ytrues.append(Y[0].cpu().numpy())
              stacks.append([z.cpu().numpy() for z in XX])
              
    from PIL import Image
    from matplotlib import cm
    
    for i in range(3):        
        x = np.moveaxis(Yhats[i], 0,2 )
        xt = np.moveaxis(Ytrues[i], 0,2 )
        if args.image_channels=='3channels':
            img = Image.fromarray(normalize8(x), 'RGB')
        else:
            x = x[:, :, 0]
            xt = xt[:, :, 0]
            img = Image.fromarray(x* 255)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # img.save('image/teste'+str(i)+'_epoch_'+str(epochv)+'.png')
        my_path = "img_res/"+args.method+args.image_channels+"_jitter_"+str(args.jitter)+args.comment+"/"+str(args.fold)+"/"
        os.makedirs(my_path, exist_ok=True)
        img.save(my_path+'PRED_'+str(i)+'.png')
        if args.image_channels=='3channels':
            imgt = Image.fromarray(normalize8(xt), 'RGB')
        else:
            imgt = Image.fromarray(xt* 255)
        if imgt.mode != 'RGB':
            imgt = imgt.convert('RGB')
        imgt.save(my_path+'GT_'+str(i)+'.png')

def loadModel(image_channel):
    model = models.EDOF_CNN_pack_43()
    if (args.comment == 'grayscale'):
        model.load_state_dict(torch.load("dataset-fraunhofer_elastic_only-image_size-512-method-"+args.method+"-Z-"+str(args.Z)+"-fold-"+str(args.fold)+"-epochs-"+str(args.epochs)+"-batchsize-"+str(args.batchsize)+"-lr-0.001-cudan-0-image_channels-"+image_channel+"-automate-1-augmentation-0-augmentLevel-0-rotate-0-hflip-0-ssim-0"+".pth"))
    else:
        model.load_state_dict(torch.load("dataset-fraunhofer_elastic_only-image_size-512-method-"+args.method+"-Z-"+str(args.Z)+"-fold-"+str(args.fold)+"-epochs-"+str(args.epochs)+"-batchsize-"+str(args.batchsize)+"-lr-0.001-cudan-0-image_channels-"+image_channel+"-automate-1-augmentation-0-augmentLevel-0-rotate-0-hflip-0-ssim-0"+"-jitter-"+str(args.jitter)+"-comment-"+str(args.comment)+".pth"))
    model = model.to(device)
    return model

def proceed():
    prefix = '-'.join(f'{k}-{v}' for k, v in vars(args).items())
    print(prefix)
    ############################# data loaders #######################################

    ts_ds = dataset.Dataset('test', test_transform, args.dataset, args.Z, args.fold, test_transformY, test_common_transform)

    #to view images
    tst = DataLoader(ts_ds, 1,False,  pin_memory=True)


    if args.method=='EDOF_CNN_max':
        model = models.EDOF_CNN_max()
    elif args.method=='EDOF_CNN_3D':
        model = models.EDOF_CNN_3D(args.Z)
    elif args.method=='EDOF_CNN_backbone':
        model = models.EDOF_CNN_backbone()
    elif args.method=='EDOF_CNN_fast':
        model = models.EDOF_CNN_fast()
    elif args.method=='EDOF_CNN_RGB':
        model = models.EDOF_CNN_RGB()
    elif args.method=='EDOF_CNN_pairwise':
        model = models.EDOF_CNN_pairwise()
    elif args.method=='EDOF_CNN_pack':
        model = models.EDOF_CNN_pack()
    elif args.method=='PackNet01':
        model = models.PackNet01()
    elif args.method=='EDOF_CNN_pack_02':
        model = models.EDOF_CNN_pack_02()
    elif args.method=='EDOF_CNN_pack_03':
        model = models.EDOF_CNN_pack_03()
    elif args.method=='EDOF_CNN_pack_04':
        model = models.EDOF_CNN_pack_04()
    elif args.method=='EDOF_CNN_pack_05':
        model = models.EDOF_CNN_pack_05()
    elif args.method=='EDOF_CNN_pack_06':
        model = models.EDOF_CNN_pack_06()
    elif args.method=='EDOF_CNN_pack_07':
        model = models.EDOF_CNN_pack_07()
    elif args.method=='EDOF_CNN_pack_08':
        model = models.EDOF_CNN_pack_08()
    elif args.method=='EDOF_CNN_pack_09':
        model = models.EDOF_CNN_pack_09()
    elif args.method=='EDOF_CNN_pack_10':
        model = models.EDOF_CNN_pack_10()
    elif args.method=='EDOF_CNN_pack_11':
        model = models.EDOF_CNN_pack_11()
    elif args.method=='EDOF_CNN_pack_12':
        model = models.EDOF_CNN_pack_12()
    elif args.method=='EDOF_CNN_pack_13':
        model = models.EDOF_CNN_pack_13()
    elif args.method=='EDOF_CNN_pack_14':
        model = models.EDOF_CNN_pack_14()
    elif args.method=='EDOF_CNN_pack_15':
        model = models.EDOF_CNN_pack_15()
    elif args.method=='EDOF_CNN_pack_16':
        model = models.EDOF_CNN_pack_16()
    elif args.method=='EDOF_CNN_pack_17':
        model = models.EDOF_CNN_pack_17()
    elif args.method=='EDOF_CNN_pack_18':
        model = models.EDOF_CNN_pack_18()
    elif args.method=='EDOF_CNN_pack_19':
        model = models.EDOF_CNN_pack_19()
    elif args.method=='EDOF_CNN_pack_20':
        model = models.EDOF_CNN_pack_20()
    elif args.method=='EDOF_CNN_pack_21':
        model = models.EDOF_CNN_pack_21()
    elif args.method=='EDOF_CNN_pack_22':
        model = models.EDOF_CNN_pack_22()
    elif args.method=='EDOF_CNN_pack_23':
        model = models.EDOF_CNN_pack_23()
    elif args.method=='EDOF_CNN_pack_24':
        model = models.EDOF_CNN_pack_24()
    elif args.method=='EDOF_CNN_pack_25':
        model = models.EDOF_CNN_pack_25()
    elif args.method=='EDOF_CNN_pack_26':
        model = models.EDOF_CNN_pack_26()
    elif args.method=='EDOF_CNN_pack_27':
        model = models.EDOF_CNN_pack_27()
    elif args.method=='EDOF_CNN_pack_28':
        model = models.EDOF_CNN_pack_28()
    elif args.method=='EDOF_CNN_pack_29':
        model = models.EDOF_CNN_pack_29()
    elif args.method=='EDOF_CNN_pack_30':
        model = models.EDOF_CNN_pack_30()
    elif args.method=='EDOF_CNN_pack_31':
        model = models.EDOF_CNN_pack_31()
    elif args.method=='EDOF_CNN_pack_32':
        model = models.EDOF_CNN_pack_32()
    elif args.method=='EDOF_CNN_pack_33':
        model = models.EDOF_CNN_pack_33()
    elif args.method=='EDOF_CNN_pack_34':
        model = models.EDOF_CNN_pack_34()
    elif args.method=='EDOF_CNN_pack_35':
        model = models.EDOF_CNN_pack_35()
    elif args.method=='EDOF_CNN_pack_36':
        model = models.EDOF_CNN_pack_36()
    elif args.method=='EDOF_CNN_pack_37':
        model = models.EDOF_CNN_pack_37()
    elif args.method=='EDOF_CNN_pack_38':
        model = models.EDOF_CNN_pack_38()
    elif args.method=='EDOF_CNN_pack_39':
        model = models.EDOF_CNN_pack_39()
    elif args.method=='EDOF_CNN_pack_40':
        model = models.EDOF_CNN_pack_40()
    elif args.method=='EDOF_CNN_pack_41':
        model = models.EDOF_CNN_pack_41()
    elif args.method=='EDOF_CNN_pack_42':
        model = models.EDOF_CNN_pack_42()
    elif args.method=='EDOF_CNN_pack_43' and args.image_channels=='3channels':
        model_red = loadModel("red")
        model_green = loadModel("green")
        model_blue = loadModel("blue")
        model = models.EDOF_CNN_pack_ensemble(model_red,model_green,model_blue)
    elif args.method=='EDOF_CNN_pack_43':
        model = models.EDOF_CNN_pack_43()
    else: 
        model = models.EDOF_CNN_concat()


    if args.image_channels!='3channels':
        model.load_state_dict(torch.load("dataset-fraunhofer_elastic_only-image_size-512-method-"+args.method+"-Z-"+str(args.Z)+"-fold-"+str(args.fold)+"-epochs-"+str(args.epochs)+"-batchsize-"+str(args.batchsize)+"-lr-0.001-cudan-0-image_channels-"+args.image_channels+"-automate-1-augmentation-0-augmentLevel-0-rotate-0-hflip-0-ssim-0.pth"))
    model = model.to(device)

    view_images(model=model, tst=tst)

# model_list = ['EDOF_CNN_pack_31','EDOF_CNN_pack_32','EDOF_CNN_pack_33']
# model_list = ['EDOF_CNN_pack_41','EDOF_CNN_pack_42']
model_list = ['EDOF_CNN_pack_43']
# model_list = ['EDOF_CNN_fast']
fold_list = [0,1,2,3,4]
for this_model in model_list:
    for this_fold in fold_list:
        args.method = this_model
        args.fold = this_fold
        proceed()