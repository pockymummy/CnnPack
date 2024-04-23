# =============================================================================
# Code to train EDoF CNNS models
# =============================================================================

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['cervix93','fraunhofer','fraunhofer_separate','fraunhofer_elastic','fraunhofer_elastic_only', 'fraunhofer_raw'], default='cervix93_zstacks')
parser.add_argument('--image_size', choices=[512,640], default=512)
parser.add_argument('--method', choices=[
    'EDOF_CNN_max','EDOF_CNN_3D','EDOF_CNN_concat','EDOF_CNN_backbone','EDOF_CNN_fast','EDOF_CNN_RGB','EDOF_CNN_pairwise','EDOF_CNN_pack', 'PackNet01'], default='EDOF_CNN_fast')
parser.add_argument('--Z', choices=[3,5,7,9], type=int, default=5)
parser.add_argument('--fold', type=int, choices=range(5),default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--cudan', type=int, default=0)
parser.add_argument('--image_channels', choices=['rgb','grayscale'], default='grayscale')
parser.add_argument('--automate', choices=[0,1], default=1)
parser.add_argument('--augmentation', choices=[0,1], default=0)
parser.add_argument('--augmentLevel', choices=[0,1], default=0)
parser.add_argument('--rotate', choices=[0,1,2,3,4], default=0)
parser.add_argument('--hflip', choices=[0,1], default=0)
parser.add_argument('--ssim', choices=[0,1], default=0)
parser.add_argument('--jitter', choices=[0,1], default=0)
parser.add_argument('--comment', choices=['',''], default='colortransferFromInput')
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
from skimage.exposure import match_histograms


device = torch.device('cuda:'+str(args.cudan) if torch.cuda.is_available() else 'cpu')

#define transforms if rgb or not
test_transform=dataset.val_transforms
train_transformY=dataset.val_transforms
test_transformY=dataset.val_transforms

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

# print some metrics 
def predict_metrics(data, model):
    model.eval()
    Phat = []
    Y_true=[]
    input_list =[]
    with torch.no_grad():
        for XX, Y in data:
            XX = [X.to(device, torch.float) for X in XX]
            Y = Y.to(device, torch.float)
            Yhat = model(XX)
            # print(Yhat.shape)
            Phat += list(Yhat.cpu().numpy())
            Y_true += list(Y.cpu().numpy())
            numpy_list= []
            for x in XX:
                numpy_list+= list(x.cpu().numpy())
            input_list.append(numpy_list)
    return Y_true, Phat, input_list

def normalize8(I):
  mn = I.min()
  mx = I.max()

  mx -= mn

  I = ((I - mn)/mx) * 255
  return I.astype(np.uint8)

def proceed():
    prefix = '-'.join(f'{k}-{v}' for k, v in vars(args).items())
    print(prefix)
    ############################# data loaders #######################################

    ts_ds = dataset.Dataset('test', dataset.val_transforms, args.dataset, args.Z, args.fold, dataset.val_transforms, dataset.common_transform_empty)

    if (args.method == 'EDOF_CNN_fast'):
        model = models.EDOF_CNN_fast()
    else:
        model = models.EDOF_CNN_pack_43()

    model.load_state_dict(torch.load("dataset-"+args.dataset+"-image_size-512-method-"+args.method+"-Z-"+str(args.Z)+"-fold-"+str(args.fold)+"-epochs-"+str(args.epochs)+"-batchsize-"+str(args.batchsize)+"-lr-0.001-cudan-0-image_channels-"+str(args.image_channels)+"-automate-1-augmentation-0-augmentLevel-0-rotate-0-hflip-0-ssim-0-jitter-0"+"-comment-"+".pth"))
    model = model.to(device)

    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, normalized_root_mse 


    data_test = DataLoader(ts_ds, 1,False,  pin_memory=True)
    Y_true, Phat, X_list = predict_metrics(data_test,model)

    import colortrans

    if (args.comment in ['uintimg','colortransfer','colortransferFromInput']):
        # Phat = list(map(normalize8,Phat))
        # Y_true = list(map(normalize8,Y_true))
        if (args.comment in ['colortransfer','colortransferFromInput']):
            color_transfered = []
            if (args.comment == 'colortransfer'):
                ref_list = Y_true
            elif (args.comment == 'colortransferFromInput'):
                temp_list = []
                for x in X_list:
                #     my_mean = np.array(x)
                #     my_mean = np.mean(my_mean, axis= 0, dtype=np.uint8)
                    temp_list.append(x[0])
                # temp_list = list(map(normalize8,temp_list))
                ref_list = temp_list
            for pred, ref in zip(Phat, ref_list):
                pred_reshape = np.moveaxis(pred,0,2)
                ref_reshape = np.moveaxis(ref,0,2)
                matched_histogram = match_histograms(pred_reshape, ref_reshape,channel_axis=-1)
                # col_transfered = colortrans.transfer_lhm(pred_reshape, ref_reshape)
                resume_shape = np.moveaxis(matched_histogram,2,0)
                color_transfered.append(resume_shape)
            Phat = color_transfered

    mse = np.mean([mean_squared_error(Y_true[i], Phat[i]) for i in range(len(Y_true))])
    rmse = np.mean([normalized_root_mse(Y_true[i], Phat[i]) for i in range(len(Y_true))])
    # if (args.comment in ['uintimg', 'colortransfer','colortransferFromInput']):
    #     ssim =np.mean([ssim(Y_true[i], Phat[i],channel_axis=0,data_range=255) for i in range(len(Y_true))]) 
    # else:
    ssim =np.mean([ssim(Y_true[i], Phat[i],channel_axis=0,data_range=1) for i in range(len(Y_true))]) 
    psnr =np.mean([peak_signal_noise_ratio(Y_true[i], Phat[i]) for i in range(len(Y_true))])




    f = open('my_results_fixed_3/'+ str(prefix)+'.txt', 'w')
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
    model_list = ['EDOF_CNN_fast']
    # 38 to 34 is not yet
    fold_list = [0,1,2,3,4]
    for this_model in model_list:
        for this_fold in fold_list:
            args.method = this_model
            args.fold = this_fold
            proceed()