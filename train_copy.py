# =============================================================================
# Code to train EDoF CNNS models
# =============================================================================

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['cervix93','fraunhofer','fraunhofer_separate','fraunhofer_elastic','fraunhofer_elastic_only', 'fraunhofer_raw', 'cervix93_zstacks'], default='cervix93_zstacks')
parser.add_argument('--image_size', choices=[512,640], default=512)
parser.add_argument('--method', choices=[
    'EDOF_CNN_max','EDOF_CNN_3D','EDOF_CNN_concat','EDOF_CNN_backbone','EDOF_CNN_fast','EDOF_CNN_RGB','EDOF_CNN_pairwise','EDOF_CNN_pack', 'PackNet01'], default='EDOF_CNN_pack_43')
parser.add_argument('--Z', choices=[3,5,7,9], type=int, default=5)
parser.add_argument('--fold', type=int, choices=range(5),default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--cudan', type=int, default=0)
parser.add_argument('--image_channels', choices=['rgb','grayscale'], default='blue')
parser.add_argument('--automate', choices=[0,1], default=1)
parser.add_argument('--augmentation', choices=[0,1], default=0)
parser.add_argument('--augmentLevel', choices=[0,1], default=0)
parser.add_argument('--rotate', choices=[0,1,2,3,4], default=0)
parser.add_argument('--hflip', choices=[0,1], default=0)
parser.add_argument('--ssim', choices=[0,1], default=0)
parser.add_argument('--jitter', choices=[0,1], default=0)
parser.add_argument('--comment', choices=['',''], default='')
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
from torchvision import transforms

device = torch.device('cuda:'+str(args.cudan) if torch.cuda.is_available() else 'cpu')

#define transforms if rgb or not
if args.image_channels=='rgb':
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
elif args.image_channels=='red':
    if args.comment=='rgbjitter':
        train_transform=dataset.transforms_PIL_preprocess
        test_transform=dataset.transforms_PIL_preprocess
        train_transformY=dataset.transforms_PIL_preprocess
        test_transformY=dataset.transforms_PIL_preprocess
        if args.jitter==2:
            common_transform=transforms.Compose(dataset.transform_jitter_2 + dataset.transforms_red_pil + dataset.transforms_to_tensor)
            test_common_transform = transforms.Compose(dataset.transforms_red_pil + dataset.transforms_to_tensor)
        elif args.jitter==3:
            common_transform=transforms.Compose(dataset.transform_jitter_3 + dataset.transforms_red_pil + dataset.transforms_to_tensor)
            test_common_transform = transforms.Compose(dataset.transforms_red_pil + dataset.transforms_to_tensor)
    else:
        train_transform=dataset.aug_transforms_red
        test_transform=dataset.aug_transforms_red
        train_transformY=dataset.aug_transforms_red
        test_transformY=dataset.aug_transforms_red
        if args.jitter==1:
            common_transform=dataset.common_transform_jitter_1
            test_common_transform=dataset.common_transform_empty
        else:
            common_transform=dataset.common_transform_empty
            test_common_transform=dataset.common_transform_empty
elif args.image_channels=='green':
    if args.comment=='rgbjitter':
        train_transform=dataset.transforms_PIL_preprocess
        test_transform=dataset.transforms_PIL_preprocess
        train_transformY=dataset.transforms_PIL_preprocess
        test_transformY=dataset.transforms_PIL_preprocess
        if args.jitter==2:
            common_transform=transforms.Compose(dataset.transform_jitter_2 + dataset.transforms_green_pil + dataset.transforms_to_tensor)
            test_common_transform = transforms.Compose(dataset.transforms_green_pil + dataset.transforms_to_tensor)
        elif args.jitter==3:
            common_transform=transforms.Compose(dataset.transform_jitter_3 + dataset.transforms_green_pil + dataset.transforms_to_tensor)
            test_common_transform = transforms.Compose(dataset.transforms_green_pil + dataset.transforms_to_tensor)
    else:
        train_transform=dataset.aug_transforms_green
        test_transform=dataset.aug_transforms_green
        train_transformY=dataset.aug_transforms_green
        test_transformY=dataset.aug_transforms_green
        common_transform=dataset.common_transform_empty
        if args.jitter==1:
            common_transform=dataset.common_transform_jitter_1
            test_common_transform=dataset.common_transform_empty
        else:
            common_transform=dataset.common_transform_empty
            test_common_transform=dataset.common_transform_empty
elif args.image_channels=='blue':
    if args.comment=='rgbjitter':
        train_transform=dataset.transforms_PIL_preprocess
        test_transform=dataset.transforms_PIL_preprocess
        train_transformY=dataset.transforms_PIL_preprocess
        test_transformY=dataset.transforms_PIL_preprocess
        if args.jitter==2:
            common_transform=transforms.Compose(dataset.transform_jitter_2 + dataset.transforms_blue_pil + dataset.transforms_to_tensor)
            test_common_transform = transforms.Compose(dataset.transforms_blue_pil + dataset.transforms_to_tensor)
        elif args.jitter==3:    
            common_transform=transforms.Compose(dataset.transform_jitter_3 + dataset.transforms_blue_pil + dataset.transforms_to_tensor)
            test_common_transform = transforms.Compose(dataset.transforms_blue_pil + dataset.transforms_to_tensor)
    else:
        train_transform=dataset.aug_transforms_blue
        test_transform=dataset.aug_transforms_blue
        train_transformY=dataset.aug_transforms_blue
        test_transformY=dataset.aug_transforms_blue
        common_transform=dataset.common_transform_empty
        if args.jitter==1:
            common_transform=dataset.common_transform_jitter_1
            test_common_transform=dataset.common_transform_empty
        else:
            common_transform=dataset.common_transform_empty
            test_common_transform=dataset.common_transform_empty
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
    test_common_transform=dataset.common_transform_empty

def view_images(epochv,model,tst):
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
        if args.epochs==200:
            stack = stacks[i]
            for s in range(args.Z):
                stack0 = Image.fromarray(stack[s][0,0,:,:]* 255)
                if stack0.mode != 'RGB':
                    stack0 = stack0.convert('RGB')
                stack0.save('teste_'+str(i)+'_stack_'+str(s)+'.png')
        
        x = np.moveaxis(Yhats[i], 0,2 )
        xt = np.moveaxis(Ytrues[i], 0,2 )
        x = x[:, :, 0]
        xt = xt[:, :, 0]
        # img = Image.fromarray(np.uint8(x*255), 'RGB')
        img = Image.fromarray(x* 255)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # img.save('image/teste'+str(i)+'_epoch_'+str(epochv)+'.png')
        img.save('PRED_'+str(i)+'.png')
        # imgt = Image.fromarray(np.uint8(xt*255), 'RGB')
        imgt = Image.fromarray(xt* 255)
        if imgt.mode != 'RGB':
            imgt = imgt.convert('RGB')
        imgt.save('GT_'+str(i)+'.png')



def test(val, model):
    model.eval()
    avg_loss_val = 0
    with torch.no_grad():
        for XX, Y in val:
            XX = [X.to(device, torch.float) for X in XX]
            Y = Y.to(device, torch.float)
            Yhat = model(XX)
            loss = model.loss(Yhat, Y.to(torch.float))
            avg_loss_val += loss / len(val)
    return avg_loss_val



def train(tr, val, model, opt, scheduler, tst, epochs=args.epochs, verbose=True):
    for epoch in range(epochs):
        if verbose:
            print(f'* Epoch {epoch+1}/{args.epochs}')
        tic = time()
        model.train()
        avg_acc = 0
        avg_loss = 0
        for XX, Y in tr:
            XX = [X.to(device, torch.float) for X in XX]
            Y = Y.to(device, torch.float)
            opt.zero_grad()
            Yhat = model(XX)
            loss = model.loss(Yhat, Y)
            if args.ssim == 1:
                loss.backward(retain_graph=True)
                ssim_loss = models.ssim_loss(Yhat, Y)
                ssim_loss.backward()
            else:
                loss.backward()
            opt.step()
            avg_loss += loss / len(tr)

        dt = time() - tic
        out = ' - %ds - Loss: %f' % (dt, avg_loss)
        if val:
            model.eval()
            out += ', Test loss: %f' % test(val, model)
        if verbose:
            print(out)
        scheduler.step(avg_loss)
        
        #uncomment to see the examples
        view_images(epoch, model=model, tst=tst)

def proceed():
    prefix = '-'.join(f'{k}-{v}' for k, v in vars(args).items())
    print(prefix)
    ############################# data loaders #######################################

    tr_ds = dataset.Dataset('train', train_transform, args.dataset, args.Z, args.fold, train_transformY, common_transform)
    tr = DataLoader(tr_ds, args.batchsize, True,  pin_memory=True)
    ts_ds = dataset.Dataset('test', test_transform, args.dataset, args.Z, args.fold, test_transformY, test_common_transform)
    ts = DataLoader(ts_ds, args.batchsize,False,  pin_memory=True)

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
    # elif args.method=='EDOF_CNN_pack_02':
    #     model = models.EDOF_CNN_pack_02()
    # elif args.method=='EDOF_CNN_pack_03':
    #     model = models.EDOF_CNN_pack_03()
    # elif args.method=='EDOF_CNN_pack_04':
    #     model = models.EDOF_CNN_pack_04()
    # elif args.method=='EDOF_CNN_pack_05':
    #     model = models.EDOF_CNN_pack_05()
    # elif args.method=='EDOF_CNN_pack_06':
    #     model = models.EDOF_CNN_pack_06()
    # elif args.method=='EDOF_CNN_pack_07':
    #     model = models.EDOF_CNN_pack_07()
    # elif args.method=='EDOF_CNN_pack_08':
    #     model = models.EDOF_CNN_pack_08()
    # elif args.method=='EDOF_CNN_pack_09':
    #     model = models.EDOF_CNN_pack_09()
    # elif args.method=='EDOF_CNN_pack_10':
    #     model = models.EDOF_CNN_pack_10()
    # elif args.method=='EDOF_CNN_pack_11':
    #     model = models.EDOF_CNN_pack_11()
    # elif args.method=='EDOF_CNN_pack_12':
    #     model = models.EDOF_CNN_pack_12()
    # elif args.method=='EDOF_CNN_pack_13':
    #     model = models.EDOF_CNN_pack_13()
    # elif args.method=='EDOF_CNN_pack_14':
    #     model = models.EDOF_CNN_pack_14()
    # elif args.method=='EDOF_CNN_pack_15':
    #     model = models.EDOF_CNN_pack_15()
    # elif args.method=='EDOF_CNN_pack_16':
    #     model = models.EDOF_CNN_pack_16()
    # elif args.method=='EDOF_CNN_pack_17':
    #     model = models.EDOF_CNN_pack_17()
    # elif args.method=='EDOF_CNN_pack_18':
    #     model = models.EDOF_CNN_pack_18()
    # elif args.method=='EDOF_CNN_pack_19':
    #     model = models.EDOF_CNN_pack_19()
    # elif args.method=='EDOF_CNN_pack_20':
    #     model = models.EDOF_CNN_pack_20()
    # elif args.method=='EDOF_CNN_pack_21':
    #     model = models.EDOF_CNN_pack_21()
    # elif args.method=='EDOF_CNN_pack_22':
    #     model = models.EDOF_CNN_pack_22()
    # elif args.method=='EDOF_CNN_pack_23':
    #     model = models.EDOF_CNN_pack_23()
    # elif args.method=='EDOF_CNN_pack_24':
    #     model = models.EDOF_CNN_pack_24()
    # elif args.method=='EDOF_CNN_pack_25':
    #     model = models.EDOF_CNN_pack_25()
    # elif args.method=='EDOF_CNN_pack_26':
    #     model = models.EDOF_CNN_pack_26()
    # elif args.method=='EDOF_CNN_pack_27':
    #     model = models.EDOF_CNN_pack_27()
    # elif args.method=='EDOF_CNN_pack_28':
    #     model = models.EDOF_CNN_pack_28()
    # elif args.method=='EDOF_CNN_pack_29':
    #     model = models.EDOF_CNN_pack_29()
    # elif args.method=='EDOF_CNN_pack_30':
    #     model = models.EDOF_CNN_pack_30()
    # elif args.method=='EDOF_CNN_pack_31':
    #     model = models.EDOF_CNN_pack_31()
    # elif args.method=='EDOF_CNN_pack_32':
    #     model = models.EDOF_CNN_pack_32()
    # elif args.method=='EDOF_CNN_pack_33':
    #     model = models.EDOF_CNN_pack_33()
    # elif args.method=='EDOF_CNN_pack_34':
    #     model = models.EDOF_CNN_pack_34()
    # elif args.method=='EDOF_CNN_pack_35':
    #     model = models.EDOF_CNN_pack_35()
    # elif args.method=='EDOF_CNN_pack_36':
    #     model = models.EDOF_CNN_pack_36()
    # elif args.method=='EDOF_CNN_pack_37':
    #     model = models.EDOF_CNN_pack_37()
    # elif args.method=='EDOF_CNN_pack_38':
    #     model = models.EDOF_CNN_pack_38()
    # elif args.method=='EDOF_CNN_pack_39':
    #     model = models.EDOF_CNN_pack_39()
    # elif args.method=='EDOF_CNN_pack_40':
    #     model = models.EDOF_CNN_pack_40()
    # elif args.method=='EDOF_CNN_pack_41':
    #     model = models.EDOF_CNN_pack_41()
    # elif args.method=='EDOF_CNN_pack_42':
    #     model = models.EDOF_CNN_pack_42()
    elif args.method=='EDOF_CNN_pack_43':
        model = models.EDOF_CNN_pack_43()
    # elif args.method=='EDOF_CNN_pack_44':
    #     model = models.EDOF_CNN_pack_44()
    else: 
        model = models.EDOF_CNN_concat()



    # model.load_state_dict(torch.load('results\\dataset-cervix93-image_size-512-method-EDOF_CNN_fast-Z-5-fold-0-epochs-200-batchsize-4-lr-0.001-cudan-1-image_channels-grayscale.pth'))
    model = model.to(device)


    opt = optim.Adam(model.parameters(), args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True,patience=5)
    train(tr, ts,model=model,opt=opt, scheduler=scheduler, tst=tst)


    torch.save(model.state_dict(), str(prefix)+'.pth')

    
    #print some metrics 
    def predict_metrics(data, model):
        model.eval()
        Phat = []
        Y_true=[]
        with torch.no_grad():
            for XX, Y in data:
                XX = [X.to(device, torch.float) for X in XX]
                Y = Y.to(device, torch.float)
                Yhat = model(XX)
                Phat += list(Yhat.cpu().numpy())
                Y_true += list(Y.cpu().numpy())
        return Y_true, Phat



    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, normalized_root_mse 


    data_test = DataLoader(ts_ds, 1,False,  pin_memory=True)
    Y_true, Phat = predict_metrics(data_test,model=model)

    mse = np.mean([mean_squared_error(Y_true[i], Phat[i]) for i in range(len(Y_true))])
    rmse = np.mean([normalized_root_mse(Y_true[i], Phat[i]) for i in range(len(Y_true))])
    ssim =np.mean([ssim(Y_true[i], Phat[i],channel_axis=0,data_range=1) for i in range(len(Y_true))]) 
    psnr =np.mean([peak_signal_noise_ratio(Y_true[i], Phat[i]) for i in range(len(Y_true))]) 



    f = open('my_results_fixed_3/'+ str(prefix)+'.txt', 'a+')
    f.write('\n\nModel:'+str(prefix)+
        ' \nMSE:'+ str(mse)+
        ' \nRMSE:'+ str(rmse)+
        ' \nSSIM:'+str(ssim)+
        ' \nPSNR:'+ str(psnr))
    f.close()

    print('my_results_fixed_3/'+ str(prefix)+'.txt')

if args.automate == 0:
    proceed()
else:
    # model_list = ['EDOF_CNN_fast']
    # model_list = ['EDOF_CNN_pack','EDOF_CNN_pack_02','EDOF_CNN_fast','EDOF_CNN_max','EDOF_CNN_3D','EDOF_CNN_pairwise']
    # model_list = ['EDOF_CNN_fast','EDOF_CNN_max','EDOF_CNN_3D','EDOF_CNN_pairwise']
    # model_list = ['EDOF_CNN_pack_43','EDOF_CNN_pack_42','EDOF_CNN_pack_41','EDOF_CNN_pack_40','EDOF_CNN_pack_39','EDOF_CNN_pack_38','EDOF_CNN_pack_37','EDOF_CNN_pack_36','EDOF_CNN_pack_35','EDOF_CNN_pack_34']
    # model_list = ['EDOF_CNN_pack_41','EDOF_CNN_pack_40','EDOF_CNN_pack_39']
    # model_list = ['EDOF_CNN_pack_32','EDOF_CNN_pack_33','EDOF_CNN_pack_34','EDOF_CNN_pack_35']
    # model_list = ['EDOF_CNN_pack_38','EDOF_CNN_pack_37','EDOF_CNN_pack_36','EDOF_CNN_pack_35','EDOF_CNN_pack_34']
    # model_list = ['EDOF_CNN_pack_20','EDOF_CNN_pack_21','EDOF_CNN_pack_22']
    # model_list = ['EDOF_CNN_pack_26','EDOF_CNN_pack_27']
    model_list = ['EDOF_CNN_pack_43']
    # 38 to 34 is not yet
    fold_list = [0,1,2,3,4]
    for this_model in model_list:
        for this_fold in fold_list:
            args.method = this_model
            args.fold = this_fold
            proceed()




# def test_cyto(path_f='test_data_aligned',img_size=640):
#     cyto_ds = dataset.Dataset_folder(dataset.val_transforms, path_f , args.Z,img_size)
#     cyto_ts = DataLoader(cyto_ds, 1 ,False,  pin_memory=True)
#     model.eval()
#     avg_loss_val = 0
#     with torch.no_grad():
#         for XX in tqdm(cyto_ts):
#             print(XX)
#             XX = [X.to(device, torch.float) for X in XX]
#             Yhat = model(XX)
#     final_edf=Yhat.cpu().numpy()
#     img=Image.fromarray(final_edf[0,0,:,:]* 255)
#     if img.mode != 'RGB':
#         img = img.convert('RGB')
#     img.save('teste_cyto.png')