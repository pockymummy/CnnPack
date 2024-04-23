# =============================================================================
# Models architectures
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchgeometry.losses import DiceLoss, ssim
from utils import ConvLayer, ResidualLayer, DeconvLayer
from einops import rearrange
from utils_files import pytorch_ssim
import kornia.losses


mse = nn.MSELoss()
mae = nn.L1Loss()
ssim_loss = pytorch_ssim.SSIM()
l1 = nn.L1Loss(reduction='sum')

from torch import nn
import torch
from piq import TVLoss

lossTV = TVLoss()

def tv_loss(c):
    x = c[:,:,1:,:] - c[:,:,:-1,:]
    y = c[:,:,:,1:] - c[:,:,:,:-1]
    loss = torch.sum(torch.abs(x)) + torch.sum(torch.abs(y))
    return loss

def compute_total_variation_loss(img, weight):      
    tv_h = ((img[:,:,1:,:] - img[:,:,:-1,:]).pow(2)).sum()
    tv_w = ((img[:,:,:,1:] - img[:,:,:,:-1]).pow(2)).sum()    
    return weight * (tv_h + tv_w)


class EDOF_CNN_fast(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_fast, self).__init__()        
        self.encoder = nn.Sequential(    
            ConvLayer(1, 32, 3, 1),
            ConvLayer(32, 64, 3, 2))
        
        self.residual = nn.Sequential(            
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1))
            
        self.decoder = nn.Sequential( 
            DeconvLayer(64, 32, 3, 1),
            DeconvLayer(32, 16, 3, 2, activation='relu'),
            ConvLayer(16, 1, 1, 1, activation='linear'))
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_max, max_indices= torch.min(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(input_max)
        Dec = self.decoder(RS)

        return Dec
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y) 
 # + 1e-3 * tv_loss(Yhat.clone().detach())

class EDOF_CNN_max(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_max, self).__init__()        
        self.encoder = nn.Sequential(    
            ConvLayer(1, 32, 3, 1),
            ConvLayer(32, 64, 3, 2),
            ConvLayer(64, 128, 3, 2))
        
        self.residual = nn.Sequential(            
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1))
            
        self.decoder = nn.Sequential( 
            DeconvLayer(128, 64, 3, 1),
            DeconvLayer(64, 32, 3, 1),
            DeconvLayer(32, 16, 3, 2, activation='relu'),
            ConvLayer(16, 1, 1, 1, activation='linear'))
        
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_max, max_indices= torch.min(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(input_max)
        Dec = self.decoder(RS)
        return Dec
    
    def loss(self, Yhat, Y): 
        return mse(Yhat, Y)
    
    # + ssim_loss(Yhat,Y)


class EDOF_CNN_3D(nn.Module):    
    def __init__(self,Z):    
        super(EDOF_CNN_3D, self).__init__()        
        self.encoder = nn.Sequential(    
            ConvLayer(Z, 32, 9, 1), #ver quantos z stacks temos
            ConvLayer(32, 64, 3, 2),
            ConvLayer(64, 128, 3, 2)
            )
        
        self.residual = nn.Sequential(            
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1))
            
        self.decoder = nn.Sequential( 
            DeconvLayer(128, 64, 3, 1),
            DeconvLayer(64, 32, 3, 1),
            DeconvLayer(32, 16, 3, 2),
            ConvLayer(16, 1, 1, 1, activation='linear'))
        
        
    def forward(self, XX):
        XXX = torch.squeeze(torch.stack([X for X in XX], dim=2), 1)
        Enc = self.encoder(XXX)
        RS = self.residual(Enc)
        Dec = self.decoder(RS)
        return Dec
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)



class EDOF_CNN_concat(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_concat, self).__init__()        
        self.encoder = nn.Sequential(    
            ConvLayer(1, 32, 3, 1),
            ConvLayer(32, 64, 3, 2),
            ConvLayer(64, 448, 3, 2))
        
        self.residual = nn.Sequential(            
            ResidualLayer(448, 448, 3, 1),
            ResidualLayer(448, 448, 3, 1),
            ResidualLayer(448, 448, 3, 1),
            ResidualLayer(448, 448, 3, 1),
            ResidualLayer(448, 448, 3, 1))

            
        self.decoder = nn.Sequential( 
            DeconvLayer(448, 64, 3, 1),
            DeconvLayer(64, 32, 3, 1),
            DeconvLayer(32, 16, 3, 2, activation='relu'),
            ConvLayer(16, 1, 1, 1, activation='linear'))
        
        #Super resolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=2, padding_mode='replicate')
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2, padding_mode='replicate')
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_cat = torch.stack(Enc)
        input_cat = rearrange(input_cat, 'd0 d1 d2 d3 d4 -> d0 (d1 d2) d3 d4')
        RS = self.residual(input_cat)
        Dec = self.decoder(RS)
        x = F.relu(self.conv1(Dec))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)
    
class EDOF_CNN_backbone(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_backbone, self).__init__()      
        model = models.mobilenet_v2(pretrained=True)
        model = nn.Sequential(*tuple(model.children())[:-1])
        # last_dimension = torch.flatten(model(torch.randn(1, 3, 640, 640))).shape[0]

        self.encoder = nn.Sequential(    
            model)

        self.residual = nn.Sequential(
            ConvLayer(1280, 254, 1, 2),            
            ResidualLayer(254, 254, 3, 1),
            ResidualLayer(254, 254, 3, 1),
            ResidualLayer(254, 254, 3, 1),
            # ResidualLayer(128, 128, 3, 1),
            # ResidualLayer(128, 128, 3, 1),
            # ResidualLayer(128, 128, 3, 1),
            # ResidualLayer(128, 128, 3, 1),
            # ResidualLayer(128, 128, 3, 1),
            ResidualLayer(254, 254, 3, 1))
            
        self.decoder = nn.Sequential( 
            DeconvLayer(254, 128, 3, 1),
            DeconvLayer(128, 64, 3, 1),
            DeconvLayer(64, 32, 3, 1),
            DeconvLayer(32, 16, 3, 1),
            DeconvLayer(16, 8, 3, 1),
            DeconvLayer(8, 4, 3, 1),
            DeconvLayer(4, 3, 3, 2, activation='relu'),
            ConvLayer(3, 3, 1, 1, activation='linear'))
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_max, max_indices= torch.max(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(input_max)
        Dec = self.decoder(RS)
        return Dec
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y) 
    # + 1e-3 * tv_loss(Yhat.clone().detach())


class EDOF_CNN_RGB(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_RGB, self).__init__()        
        self.encoder = nn.Sequential(    
            ConvLayer(3, 32, 3, 1),
            ConvLayer(32, 64, 3, 2))
        
        self.residual = nn.Sequential(            
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1))
            
        self.decoder = nn.Sequential( 
            DeconvLayer(64, 32, 3, 1),
            DeconvLayer(32, 16, 3, 2, activation='relu'),
            ConvLayer(16, 3, 1, 1, activation='linear'))
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_max, max_indices= torch.min(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(input_max)
        Dec = self.decoder(RS)

        return Dec
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y) 



class EDOF_CNN_pairwise(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pairwise, self).__init__()        
        self.encoder = nn.Sequential(    
            ConvLayer(1, 32, 3, 1),
            ConvLayer(32, 64, 3, 2))
        
        self.encoder2 = nn.Sequential(    
            ConvLayer(64, 64, 3, 1),
            ConvLayer(64, 64, 3, 1))
        
        self.residual = nn.Sequential(            
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1))
            
        self.decoder = nn.Sequential( 
            DeconvLayer(64, 32, 3, 1),
            DeconvLayer(32, 16, 3, 2, activation='relu'),
            ConvLayer(16, 1, 1, 1, activation='linear'))
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        
        input_max0, max_indices= torch.min(torch.stack(Enc[0:1]),dim=0,keepdim=False)
        input_max2, max_indices= torch.min(torch.stack(Enc[1:3]),dim=0,keepdim=False)
        
        XXX=[input_max0, input_max2]
        
        Enc = [self.encoder2(X) for X in XXX]
        input_max, max_indices= torch.min(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(input_max)
        Dec = self.decoder(RS)

        return Dec
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y) 
    
from layers01 import \
    PackLayerConv3d, UnpackLayerConv3d, Conv2D, ResidualBlock, InvDepth
    

class EDOF_CNN_pack(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack, self).__init__()    

        self.encoder = nn.Sequential(    
            Conv2D(1, 16, 5, 1),
            PackLayerConv3d(16, 5),
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 5)
            )
        
        self.residual = nn.Sequential(            
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1))
            
        # UnpackLayerConv3d(64, 32, 3)
        self.decoder = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 32, 3, 1),
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 32, 3, 1),
            ConvLayer(32, 16, 1, 1, activation='relu'),
            ConvLayer(16, 1, 1, 1, activation='linear')
            )
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_max, max_indices= torch.min(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(input_max)
        Dec = self.decoder(RS)

        return Dec
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y) 
    
class EDOF_CNN_pack_02(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_02, self).__init__()    

        self.encoder = nn.Sequential(    
            Conv2D(1, 16, 3, 1),
            PackLayerConv3d(16, 3),
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
            )
        
        self.residual = nn.Sequential(            
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1))
            
        self.decoder = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1),
            UnpackLayerConv3d(16, 16, 3),
            Conv2D(16, 16, 3, 1),
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
            )
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_max, max_indices= torch.min(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(input_max)
        Dec = self.decoder(RS)

        return Dec
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y) 
    
class EDOF_CNN_pack_03(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_03, self).__init__()    

        self.encoder = nn.Sequential(    
            ConvLayer(1, 16, 3, 1),
            PackLayerConv3d(16, 3),
            ConvLayer(16, 32, 3, 1),
            PackLayerConv3d(32, 3))
        
        self.residual = nn.Sequential(            
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1))
            
        self.decoder = nn.Sequential(
            UnpackLayerConv3d(32, 16, 3), 
            UnpackLayerConv3d(16, 16, 3),
            ConvLayer(16, 1, 1, 1, activation='linear'))
        
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_max, max_indices= torch.min(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(input_max)
        Dec = self.decoder(RS)

        return Dec
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y) 
    
class EDOF_CNN_pack_04(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_04, self).__init__()    

        self.encoder = nn.Sequential(    
            ConvLayer(1, 16, 3, 1),
            PackLayerConv3d(16, 3),
            ConvLayer(16, 32, 3, 1),
            PackLayerConv3d(32, 3))
        
        self.residual = nn.Sequential(            
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1))
            
        self.decoder = nn.Sequential(
            UnpackLayerConv3d(32, 32, 3),
            ConvLayer(32, 16, 3, 1),
            UnpackLayerConv3d(16, 16, 3),
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear'))
        
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_max, max_indices= torch.min(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(input_max)
        Dec = self.decoder(RS)

        return Dec
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

class EDOF_CNN_pack_05(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_05, self).__init__() 
        # mdf_weight_path = "/home/van/research/mdf/weights/Ds_SISR.pth"
        # from mdfloss import MDFLoss
        # self.mdf = MDFLoss(mdf_weight_path, cuda_available=True)   

        self.encoder = nn.Sequential(    
            Conv2D(1, 16, 3, 1),
            PackLayerConv3d(16, 3),
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
            )
        
        self.residual = nn.Sequential(            
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1))
            
        self.decoder = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1),
            UnpackLayerConv3d(16, 16, 3),
            Conv2D(16, 16, 3, 1),
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
            )
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_max, max_indices= torch.min(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(input_max)
        Dec = self.decoder(RS)

        return Dec
    
    def loss(self, Yhat, Y):
        # mdf_loss = self.mdf(Yhat.repeat(1, 3, 1, 1), Y.repeat(1, 3, 1, 1))
        mse_loss = mse(Yhat, Y)
        ssim = ssim_loss(Yhat,Y)
        combine_loss = 0.999*mse_loss + 0.001*ssim
        return combine_loss

class EDOF_CNN_pack_06(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_06, self).__init__()    

        self.encoder = nn.Sequential(    
            Conv2D(1, 16, 3, 1),
            PackLayerConv3d(16, 3),
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
            )

        self.residual_2 = nn.Sequential(            
            ResidualLayer(160, 160, 3, 1),
            Conv2D(160, 96, 3, 1),
            ResidualLayer(96, 96, 3, 1),
            Conv2D(96, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1))
        
        self.residual = nn.Sequential(            
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1))
            
        self.decoder = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1),
            UnpackLayerConv3d(16, 16, 3),
            Conv2D(16, 16, 3, 1),
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
            )
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        concat_filter = torch.cat(Enc, axis=1)
        fused_filter = self.residual_2(concat_filter)
        # input_max, max_indices= torch.min(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(fused_filter)
        Dec = self.decoder(RS)

        return Dec
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

class EDOF_CNN_pack_07(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_07, self).__init__()    

        self.encoder = nn.Sequential(    
            Conv2D(1, 16, 3, 1),
            PackLayerConv3d(16, 3),
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
            )
        
        self.residual = nn.Sequential(            
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1))
            
        self.decoder = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1),
            UnpackLayerConv3d(16, 16, 3),
            Conv2D(16, 16, 3, 1),
            ConvLayer(16, 1, 3, 1, activation='elu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
            )
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_max, max_indices= torch.min(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(input_max)
        Dec = self.decoder(RS)

        return Dec
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

class EDOF_CNN_pack_08(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_08, self).__init__()    

        self.encoder = nn.Sequential(    
            Conv2D(1, 16, 3, 1),
            PackLayerConv3d(16, 3),
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
            )
        
        self.residual = nn.Sequential(            
            ResidualLayer(32, 32, 3, 1),
            nn.Dropout(0.2),
            ResidualLayer(32, 32, 3, 1),
            nn.Dropout(0.2),
            ResidualLayer(32, 32, 3, 1),
            nn.Dropout(0.2),
            ResidualLayer(32, 32, 3, 1),
            nn.Dropout(0.2),
            ResidualLayer(32, 32, 3, 1))
            
        self.decoder = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1),
            UnpackLayerConv3d(16, 16, 3),
            Conv2D(16, 16, 3, 1),
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
            )
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_max, max_indices= torch.min(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(input_max)
        Dec = self.decoder(RS)

        return Dec
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y) 

class EDOF_CNN_pack_09(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_09, self).__init__()    

        self.encoder = nn.Sequential(    
            Conv2D(1, 16, 3, 1),
            PackLayerConv3d(16, 3),
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
            )
        
        self.residual = nn.Sequential(            
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1))
            
        self.decoder = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1),
            UnpackLayerConv3d(16, 16, 3),
            Conv2D(16, 16, 3, 1),
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
            )
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_max, max_indices= torch.min(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(input_max)
        Dec = self.decoder(RS)

        return Dec
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)
    
class EDOF_CNN_pack_10(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_10, self).__init__()    

        self.encoder = nn.Sequential(    
            Conv2D(1, 16, 3, 1),
            PackLayerConv3d(16, 3),
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
            )
        
        self.residual = nn.Sequential(            
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1))
            
        self.decoder = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1),
            UnpackLayerConv3d(16, 16, 3),
            Conv2D(16, 16, 3, 1),
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
            )
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_max, max_indices= torch.min(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(input_max)
        Dec = self.decoder(RS)

        return Dec
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)
    
class EDOF_CNN_pack_11(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_11, self).__init__()    

        self.encoder = nn.Sequential(    
            Conv2D(1, 8, 3, 1),
            PackLayerConv3d(8, 3),
            Conv2D(8, 16, 3, 1),
            PackLayerConv3d(16, 3)
            )
        
        self.residual = nn.Sequential(            
            ResidualLayer(16, 16, 3, 1),
            ResidualLayer(16, 16, 3, 1),
            ResidualLayer(16, 16, 3, 1),
            ResidualLayer(16, 16, 3, 1),
            ResidualLayer(16, 16, 3, 1))
            
        self.decoder = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3),
            Conv2D(16, 8, 3, 1),
            UnpackLayerConv3d(8, 8, 3),
            Conv2D(8, 8, 3, 1),
            ConvLayer(8, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
            )
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_max, max_indices= torch.min(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(input_max)
        Dec = self.decoder(RS)

        return Dec
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y) 
    
class EDOF_CNN_pack_12(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_12, self).__init__()    

        self.encoder = nn.Sequential(    
            Conv2D(1, 16, 3, 1),
            PackLayerConv3d(16, 3),
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
            )
        
        self.res_1 = ResidualLayer(32, 32, 3, 1)
        self.res_2 = ResidualLayer(32, 32, 3, 1)
        self.res_3 = ResidualLayer(32, 32, 3, 1)
        self.res_4 = ResidualLayer(32, 32, 3, 1)
        self.res_5 = ResidualLayer(32, 32, 3, 1)
            
        self.decoder = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1),
            UnpackLayerConv3d(16, 16, 3),
            Conv2D(16, 16, 3, 1),
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
            )
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_max, max_indices= torch.min(torch.stack(Enc),dim=0,keepdim=False)
        # RS = self.residual(input_max)
        rs1 = self.res_1(input_max)
        rs2 = self.res_2(rs1) 
        rs3 = self.res_3(rs2) + rs1
        rs4 = self.res_4(rs3) + rs1 + rs2
        rs5 = self.res_5(rs4) + rs1 + rs2 + rs3
        Dec = self.decoder(rs5)

        return Dec
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y) 

class EDOF_CNN_pack_13(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_13, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 16, 3, 1)
        )
        self.encoder_1 = nn.Sequential(
            PackLayerConv3d(16, 3)
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
        )
        self.residual = nn.Sequential(            
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1))
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1)
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3),
            Conv2D(16, 16, 3, 1),
            )
        self.decoder_3 = nn.Sequential(
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        input_max, max_indices= torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2)

        return decoder_3
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)
    
class EDOF_CNN_pack_14(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_14, self).__init__()    

        self.encoder = nn.Sequential(    
            Conv2D(1, 16, 3, 1,activation="relu"),
            PackLayerConv3d(16, 3,activation="relu"),
            Conv2D(16, 32, 3, 1,activation="relu"),
            PackLayerConv3d(32, 3,activation="relu")
            )
        
        self.residual = nn.Sequential(            
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1))
            
        self.decoder = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3,activation="relu"),
            Conv2D(32, 16, 3, 1,activation="relu"),
            UnpackLayerConv3d(16, 16, 3,activation="relu"),
            Conv2D(16, 16, 3, 1,activation="relu"),
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
            )
        
    def forward(self, XX):
        Enc = [self.encoder(X) for X in XX]
        input_max, max_indices= torch.min(torch.stack(Enc),dim=0,keepdim=False)
        RS = self.residual(input_max)
        Dec = self.decoder(RS)

        return Dec
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y) 
    
class EDOF_CNN_pack_15(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_15, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 8, 3, 1)
        )
        self.encoder_1 = nn.Sequential(
            PackLayerConv3d(8, 3)
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(8, 16, 3, 1),
            PackLayerConv3d(16, 3),
        )
        self.encoder_3 = nn.Sequential(
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
        )
        self.residual = nn.Sequential(            
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1))
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1)
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3),
            Conv2D(16, 8, 3, 1),
            )
        self.decoder_3 = nn.Sequential( 
            UnpackLayerConv3d(8, 8, 3),
            )
        self.decoder_4 = nn.Sequential(
            ConvLayer(8, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        encoder_3 = [self.encoder_3(X) for X in encoder_2]
        input_max, max_indices= torch.min(torch.stack(encoder_3),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip3, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2) + skip3
        decoder_4 = self.decoder_4(decoder_3)

        return decoder_4
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)
    
# residual connection
class EDOF_CNN_pack_16(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_16, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 16, 3, 1)
        )
        self.encoder_1 = nn.Sequential(
            PackLayerConv3d(16, 3)
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
        )
        self.residual = nn.Sequential(            
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1),
            ResidualLayer(32, 32, 3, 1))
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1)
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3),
            Conv2D(16, 16, 3, 1),
            )
        self.decoder_3 = nn.Sequential(
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        input_max, max_indices= torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        res = self.residual(input_max)
        decoder_1 = self.decoder_1(res) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2)

        return decoder_3
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

class EDOF_CNN_pack_17(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_17, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 4, 3, 1)
        )
        self.encoder_1 = nn.Sequential(
            PackLayerConv3d(4, 3)
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(4, 8, 3, 1),
            PackLayerConv3d(8, 3),
        )
        self.encoder_3 = nn.Sequential(
            Conv2D(8, 16, 3, 1),
            PackLayerConv3d(16, 3)
        )
        self.encoder_4 = nn.Sequential(
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
        )
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1)
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3),
            Conv2D(16, 8, 3, 1),
            )
        self.decoder_3 = nn.Sequential( 
            UnpackLayerConv3d(8, 8, 3),
            Conv2D(8, 4, 3, 1),
            )
        self.decoder_4 = nn.Sequential( 
            UnpackLayerConv3d(4, 4, 3, groupnormsize=2),
            )
        self.decoder_5 = nn.Sequential(
            ConvLayer(4, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        encoder_3 = [self.encoder_3(X) for X in encoder_2]
        encoder_4 = [self.encoder_4(X) for X in encoder_3]
        input_max, max_indices= torch.min(torch.stack(encoder_4),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_3),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip3, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip4, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2) + skip3
        decoder_4 = self.decoder_4(decoder_3) + skip4
        decoder_5 = self.decoder_5(decoder_4)

        return decoder_5
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

class EDOF_CNN_pack_18(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_18, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 8, 3, 1)
        )
        self.encoder_1 = nn.Sequential(
            PackLayerConv3d(8, 3)
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(8, 16, 3, 1),
            PackLayerConv3d(16, 3),
        )
        self.encoder_3 = nn.Sequential(
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
        )
        self.encoder_4 = nn.Sequential(
            Conv2D(32, 64, 3, 1),
            PackLayerConv3d(64, 3)
        )
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(64, 64, 3),
            Conv2D(64, 32, 3, 1)
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1),
            )
        self.decoder_3 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3),
            Conv2D(16, 8, 3, 1),
            )
        self.decoder_4 = nn.Sequential( 
            UnpackLayerConv3d(8, 8, 3, groupnormsize=2),
            )
        self.decoder_5 = nn.Sequential(
            ConvLayer(8, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        encoder_3 = [self.encoder_3(X) for X in encoder_2]
        encoder_4 = [self.encoder_4(X) for X in encoder_3]
        input_max, max_indices= torch.min(torch.stack(encoder_4),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_3),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip3, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip4, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2) + skip3
        decoder_4 = self.decoder_4(decoder_3) + skip4
        decoder_5 = self.decoder_5(decoder_4)

        return decoder_5
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)
    
class EDOF_CNN_pack_19(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_19, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 4, 3, 1)
        )
        self.encoder_1 = nn.Sequential(
            PackLayerConv3d(4, 3)
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(4, 8, 3, 1),
            PackLayerConv3d(8, 3),
        )
        self.encoder_3 = nn.Sequential(
            Conv2D(8, 16, 3, 1),
            PackLayerConv3d(16, 3)
        )
        self.encoder_4 = nn.Sequential(
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
        )
        self.encoder_5 = nn.Sequential(
            Conv2D(32, 64, 3, 1),
            PackLayerConv3d(64, 3)
        )
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(64, 64, 3),
            Conv2D(64, 32, 3, 1)
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1),
            )
        self.decoder_3 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3),
            Conv2D(16, 8, 3, 1),
            )
        self.decoder_4 = nn.Sequential( 
            UnpackLayerConv3d(8, 8, 3),
            Conv2D(8, 4, 3, 1),
            )
        self.decoder_5 = nn.Sequential( 
            UnpackLayerConv3d(4, 4, 3, groupnormsize=2),
            )
        self.decoder_6 = nn.Sequential(
            ConvLayer(4, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        encoder_3 = [self.encoder_3(X) for X in encoder_2]
        encoder_4 = [self.encoder_4(X) for X in encoder_3]
        encoder_5 = [self.encoder_5(X) for X in encoder_4]
        input_max, max_indices= torch.min(torch.stack(encoder_5),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_4),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_3),dim=0,keepdim=False)
        skip3, _ = torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip4, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip5, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2) + skip3
        decoder_4 = self.decoder_4(decoder_3) + skip4
        decoder_5 = self.decoder_5(decoder_4) + skip5
        decoder_6 = self.decoder_6(decoder_5)

        return decoder_6
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)
    
class EDOF_CNN_pack_20(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_20, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 4, 3, 1)
        )
        self.encoder_1 = nn.Sequential(
            PackLayerConv3d(4, 3)
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(4, 8, 3, 1),
            PackLayerConv3d(8, 3),
        )
        self.encoder_3 = nn.Sequential(
            Conv2D(8, 16, 3, 1),
            PackLayerConv3d(16, 3)
        )
        self.encoder_4 = nn.Sequential(
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
        )
        self.encoder_5 = nn.Sequential(
            Conv2D(32, 64, 3, 1),
            PackLayerConv3d(64, 3)
        )
        self.encoder_6 = nn.Sequential(
            Conv2D(64, 128, 3, 1),
            PackLayerConv3d(128, 3)
        )
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(128, 128, 3),
            Conv2D(128, 64, 3, 1)
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(64, 64, 3),
            Conv2D(64, 32, 3, 1),
            )
        self.decoder_3 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1),
            )
        self.decoder_4 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3),
            Conv2D(16, 8, 3, 1),
            )
        self.decoder_5 = nn.Sequential( 
            UnpackLayerConv3d(8, 8, 3),
            Conv2D(8, 4, 3, 1),
            )
        self.decoder_6 = nn.Sequential( 
            UnpackLayerConv3d(4, 4, 3, groupnormsize=2),
            )
        self.decoder_7 = nn.Sequential(
            ConvLayer(4, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        encoder_3 = [self.encoder_3(X) for X in encoder_2]
        encoder_4 = [self.encoder_4(X) for X in encoder_3]
        encoder_5 = [self.encoder_5(X) for X in encoder_4]
        encoder_6 = [self.encoder_6(X) for X in encoder_5]
        input_max, max_indices= torch.min(torch.stack(encoder_6),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_5),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_4),dim=0,keepdim=False)
        skip3, _ = torch.min(torch.stack(encoder_3),dim=0,keepdim=False)
        skip4, _ = torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip5, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip6, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2) + skip3
        decoder_4 = self.decoder_4(decoder_3) + skip4
        decoder_5 = self.decoder_5(decoder_4) + skip5
        decoder_6 = self.decoder_6(decoder_5) + skip6
        decoder_7 = self.decoder_7(decoder_6) 

        return decoder_7
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

class EDOF_CNN_pack_21(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_21, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 16, 3, 1)
        )
        self.encoder_1 = nn.Sequential(
            PackLayerConv3d(16, 3)
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
        )
        self.encoder_3 = nn.Sequential(
            Conv2D(32, 64, 3, 1),
            PackLayerConv3d(64, 3)
        )
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(64, 64, 3),
            Conv2D(64, 32, 3, 1)
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1),
            )
        self.decoder_3 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3),
            )
        self.decoder_4 = nn.Sequential(
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        encoder_3 = [self.encoder_3(X) for X in encoder_2]
        input_max, max_indices= torch.min(torch.stack(encoder_3),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip3, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2) + skip3
        decoder_4 = self.decoder_4(decoder_3)

        return decoder_4
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

class EDOF_CNN_pack_22(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_22, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 16, 3, 1)
        )
        self.encoder_1 = nn.Sequential(
            PackLayerConv3d(16, 3)
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(16, 64, 3, 1),
            PackLayerConv3d(64, 3)
        )
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(64, 64, 3),
            Conv2D(64, 16, 3, 1)
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3),
            Conv2D(16, 16, 3, 1),
            )
        self.decoder_3 = nn.Sequential(
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        input_max, max_indices= torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2)

        return decoder_3
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

class EDOF_CNN_pack_23(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_23, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 4, 3, 1)
        )
        self.encoder_1 = nn.Sequential(
            PackLayerConv3d(4, 3)
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(4, 8, 3, 1),
            PackLayerConv3d(8, 3),
        )
        self.encoder_3 = nn.Sequential(
            Conv2D(8, 16, 3, 1),
            PackLayerConv3d(16, 3)
        )
        self.encoder_4 = nn.Sequential(
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
        )
        self.encoder_5 = nn.Sequential(
            Conv2D(32, 64, 3, 1),
            PackLayerConv3d(64, 3)
        )
        self.encoder_6 = nn.Sequential(
            Conv2D(64, 128, 3, 1),
            PackLayerConv3d(128, 3)
        )
        self.encoder_7 = nn.Sequential(
            Conv2D(128, 256, 3, 1),
            PackLayerConv3d(256, 3)
        )
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(256, 256, 3),
            Conv2D(256, 128, 3, 1)
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(128, 128, 3),
            Conv2D(128, 64, 3, 1),
            )
        self.decoder_3 = nn.Sequential( 
            UnpackLayerConv3d(64, 64, 3),
            Conv2D(64, 32, 3, 1),
            )
        self.decoder_4 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1),
            )
        self.decoder_5 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3),
            Conv2D(16, 8, 3, 1),
            )
        self.decoder_6 = nn.Sequential( 
            UnpackLayerConv3d(8, 8, 3),
            Conv2D(8, 4, 3, 1),
            )
        self.decoder_7 = nn.Sequential( 
            UnpackLayerConv3d(4, 4, 3, groupnormsize=2),
            )
        self.decoder_8 = nn.Sequential(
            ConvLayer(4, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        encoder_3 = [self.encoder_3(X) for X in encoder_2]
        encoder_4 = [self.encoder_4(X) for X in encoder_3]
        encoder_5 = [self.encoder_5(X) for X in encoder_4]
        encoder_6 = [self.encoder_6(X) for X in encoder_5]
        encoder_7 = [self.encoder_7(X) for X in encoder_6]
        input_max, max_indices= torch.min(torch.stack(encoder_7),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_6),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_5),dim=0,keepdim=False)
        skip3, _ = torch.min(torch.stack(encoder_4),dim=0,keepdim=False)
        skip4, _ = torch.min(torch.stack(encoder_3),dim=0,keepdim=False)
        skip5, _ = torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip6, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip7, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2) + skip3
        decoder_4 = self.decoder_4(decoder_3) + skip4
        decoder_5 = self.decoder_5(decoder_4) + skip5
        decoder_6 = self.decoder_6(decoder_5) + skip6
        decoder_7 = self.decoder_7(decoder_6) + skip7
        decoder_8 = self.decoder_8(decoder_7) 

        return decoder_8
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)
    
class EDOF_CNN_pack_24(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_24, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 16, 3, 1)
        )
        self.encoder_1 = nn.Sequential(
            PackLayerConv3d(16, 3)
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
        )
        self.encoder_3 = nn.Sequential(
            Conv2D(32, 64, 3, 1),
            PackLayerConv3d(64, 3)
        )
        self.encoder_4 = nn.Sequential(
            Conv2D(64, 128, 3, 1),
            PackLayerConv3d(128, 3)
        )
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(128, 128, 3),
            Conv2D(128, 64, 3, 1)
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(64, 64, 3),
            Conv2D(64, 32, 3, 1)
            )
        self.decoder_3 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1),
            )
        self.decoder_4 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3),
            )
        self.decoder_5 = nn.Sequential(
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        encoder_3 = [self.encoder_3(X) for X in encoder_2]
        encoder_4 = [self.encoder_4(X) for X in encoder_3]
        input_max, max_indices= torch.min(torch.stack(encoder_4),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_3),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip3, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip4, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2) + skip3
        decoder_4 = self.decoder_4(decoder_3) + skip4
        decoder_5 = self.decoder_5(decoder_4)

        return decoder_5
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

class EDOF_CNN_pack_25(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_25, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 16, 3, 1)
        )
        self.encoder_1 = nn.Sequential(
            PackLayerConv3d(16, 3)
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(16, 64, 3, 1),
            PackLayerConv3d(64, 3)
        )
        self.encoder_3 = nn.Sequential(
            Conv2D(64, 128, 3, 1),
            PackLayerConv3d(128, 3)
        )
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(128, 128, 3),
            Conv2D(128, 64, 3, 1)
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(64, 64, 3),
            Conv2D(64, 16, 3, 1)
            )
        self.decoder_3 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3),
            Conv2D(16, 16, 3, 1),
            )
        self.decoder_4 = nn.Sequential(
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        encoder_3 = [self.encoder_3(X) for X in encoder_2]
        input_max, max_indices= torch.min(torch.stack(encoder_3),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip3, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2) + skip3
        decoder_4 = self.decoder_4(decoder_3)

        return decoder_4
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

# model from 22
# large intial kernel
class EDOF_CNN_pack_26(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_26, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 16, 5, 1)
        )
        self.encoder_1 = nn.Sequential(
            PackLayerConv3d(16, 5)
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(16, 64, 3, 1),
            PackLayerConv3d(64, 3)
        )
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(64, 64, 3),
            Conv2D(64, 16, 3, 1)
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3),
            Conv2D(16, 16, 3, 1),
            )
        self.decoder_3 = nn.Sequential(
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        input_max, max_indices= torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2)

        return decoder_3
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

# model from 21
# improvement by large kernel
class EDOF_CNN_pack_27(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_27, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 16, 5, 1)
        )
        self.encoder_1 = nn.Sequential(
            PackLayerConv3d(16, 5)
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
        )
        self.encoder_3 = nn.Sequential(
            Conv2D(32, 64, 3, 1),
            PackLayerConv3d(64, 3)
        )
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(64, 64, 3),
            Conv2D(64, 32, 3, 1)
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1),
            )
        self.decoder_3 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3),
            )
        self.decoder_4 = nn.Sequential(
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        encoder_3 = [self.encoder_3(X) for X in encoder_2]
        input_max, max_indices= torch.min(torch.stack(encoder_3),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip3, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2) + skip3
        decoder_4 = self.decoder_4(decoder_3)

        return decoder_4
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

#model from 20
#improvement by Relu
class EDOF_CNN_pack_28(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_28, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 4, 3, 1, activation="relu")
        )
        self.encoder_1 = nn.Sequential(
            PackLayerConv3d(4, 3, activation="relu")
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(4, 8, 3, 1, activation="relu"),
            PackLayerConv3d(8, 3, activation="relu"),
        )
        self.encoder_3 = nn.Sequential(
            Conv2D(8, 16, 3, 1, activation="relu"),
            PackLayerConv3d(16, 3, activation="relu")
        )
        self.encoder_4 = nn.Sequential(
            Conv2D(16, 32, 3, 1, activation="relu"),
            PackLayerConv3d(32, 3, activation="relu")
        )
        self.encoder_5 = nn.Sequential(
            Conv2D(32, 64, 3, 1, activation="relu"),
            PackLayerConv3d(64, 3, activation="relu")
        )
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(64, 64, 3, activation="relu"),
            Conv2D(64, 32, 3, 1, activation="relu")
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3, activation="relu"),
            Conv2D(32, 16, 3, 1, activation="relu"),
            )
        self.decoder_3 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3, activation="relu"),
            Conv2D(16, 8, 3, 1, activation="relu"),
            )
        self.decoder_4 = nn.Sequential( 
            UnpackLayerConv3d(8, 8, 3, activation="relu"),
            Conv2D(8, 4, 3, 1, activation="relu"),
            )
        self.decoder_5 = nn.Sequential( 
            UnpackLayerConv3d(4, 4, 3, activation="relu", groupnormsize=2),
            )
        self.decoder_6 = nn.Sequential(
            ConvLayer(4, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        encoder_3 = [self.encoder_3(X) for X in encoder_2]
        encoder_4 = [self.encoder_4(X) for X in encoder_3]
        encoder_5 = [self.encoder_5(X) for X in encoder_4]
        input_max, max_indices= torch.min(torch.stack(encoder_5),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_4),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_3),dim=0,keepdim=False)
        skip3, _ = torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip4, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip5, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2) + skip3
        decoder_4 = self.decoder_4(decoder_3) + skip4
        decoder_5 = self.decoder_5(decoder_4) + skip5
        decoder_6 = self.decoder_6(decoder_5)

        return decoder_6
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

# model from 18
# range 8-128
# improve by 64 to 128
class EDOF_CNN_pack_29(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_29, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 8, 3, 1)
        )
        self.encoder_1 = nn.Sequential(
            PackLayerConv3d(8, 3)
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(8, 16, 3, 1),
            PackLayerConv3d(16, 3),
        )
        self.encoder_3 = nn.Sequential(
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
        )
        self.encoder_4 = nn.Sequential(
            Conv2D(32, 64, 3, 1),
            PackLayerConv3d(64, 3)
        )
        self.encoder_5 = nn.Sequential(
            Conv2D(64, 128, 3, 1),
            PackLayerConv3d(128, 3)
        )
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(128, 128, 3),
            Conv2D(128, 64, 3, 1)
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(64, 64, 3),
            Conv2D(64, 32, 3, 1)
            )
        self.decoder_3 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1),
            )
        self.decoder_4 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3),
            Conv2D(16, 8, 3, 1),
            )
        self.decoder_5 = nn.Sequential( 
            UnpackLayerConv3d(8, 8, 3, groupnormsize=2),
            )
        self.decoder_6 = nn.Sequential(
            ConvLayer(8, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        encoder_3 = [self.encoder_3(X) for X in encoder_2]
        encoder_4 = [self.encoder_4(X) for X in encoder_3]
        encoder_5 = [self.encoder_5(X) for X in encoder_4]
        input_max, max_indices= torch.min(torch.stack(encoder_5),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_4),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_3),dim=0,keepdim=False)
        skip3, _ = torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip4, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip5, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2) + skip3
        decoder_4 = self.decoder_4(decoder_3) + skip4
        decoder_5 = self.decoder_5(decoder_4) + skip5
        decoder_6 = self.decoder_6(decoder_5)

        return decoder_6
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)
    
# model from 24
# range 16 - 256
# improve by 128 to 256
class EDOF_CNN_pack_30(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_30, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 16, 3, 1)
        )
        self.encoder_1 = nn.Sequential(
            PackLayerConv3d(16, 3)
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
        )
        self.encoder_3 = nn.Sequential(
            Conv2D(32, 64, 3, 1),
            PackLayerConv3d(64, 3)
        )
        self.encoder_4 = nn.Sequential(
            Conv2D(64, 128, 3, 1),
            PackLayerConv3d(128, 3)
        )
        self.encoder_5 = nn.Sequential(
            Conv2D(128, 256, 3, 1),
            PackLayerConv3d(256, 3)
        )
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(256, 256, 3),
            Conv2D(256, 128, 3, 1)
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(128, 128, 3),
            Conv2D(128, 64, 3, 1)
            )
        self.decoder_3 = nn.Sequential( 
            UnpackLayerConv3d(64, 64, 3),
            Conv2D(64, 32, 3, 1)
            )
        self.decoder_4 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1),
            )
        self.decoder_5 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3),
            )
        self.decoder_6 = nn.Sequential(
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        encoder_3 = [self.encoder_3(X) for X in encoder_2]
        encoder_4 = [self.encoder_4(X) for X in encoder_3]
        encoder_5 = [self.encoder_5(X) for X in encoder_4]
        input_max, max_indices= torch.min(torch.stack(encoder_5),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_4),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_3),dim=0,keepdim=False)
        skip3, _ = torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip4, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip5, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2) + skip3
        decoder_4 = self.decoder_4(decoder_3) + skip4
        decoder_5 = self.decoder_5(decoder_4) + skip5
        decoder_6 = self.decoder_6(decoder_5)

        return decoder_6
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

# model from 18
# try no skip connection
class EDOF_CNN_pack_31(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_31, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 8, 3, 1)
        )
        self.encoder_1 = nn.Sequential(
            PackLayerConv3d(8, 3)
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(8, 16, 3, 1),
            PackLayerConv3d(16, 3),
        )
        self.encoder_3 = nn.Sequential(
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
        )
        self.encoder_4 = nn.Sequential(
            Conv2D(32, 64, 3, 1),
            PackLayerConv3d(64, 3)
        )
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(64, 64, 3),
            Conv2D(64, 32, 3, 1)
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1),
            )
        self.decoder_3 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3),
            Conv2D(16, 8, 3, 1),
            )
        self.decoder_4 = nn.Sequential( 
            UnpackLayerConv3d(8, 8, 3, groupnormsize=2),
            )
        self.decoder_5 = nn.Sequential(
            ConvLayer(8, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        encoder_3 = [self.encoder_3(X) for X in encoder_2]
        encoder_4 = [self.encoder_4(X) for X in encoder_3]
        input_max, max_indices= torch.min(torch.stack(encoder_4),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) 
        decoder_2 = self.decoder_2(decoder_1) 
        decoder_3 = self.decoder_3(decoder_2) 
        decoder_4 = self.decoder_4(decoder_3)
        decoder_5 = self.decoder_5(decoder_4)

        return decoder_5
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

# model from 22
# improve by relu
class EDOF_CNN_pack_32(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_32, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 16, 3, 1, activation='relu')
        )
        self.encoder_1 = nn.Sequential(
            PackLayerConv3d(16, 3, activation='relu')
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(16, 64, 3, 1, activation='relu'),
            PackLayerConv3d(64, 3, activation='relu')
        )
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(64, 64, 3, activation='relu'),
            Conv2D(64, 16, 3, 1, activation='relu')
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3, activation='relu'),
            Conv2D(16, 16, 3, 1, activation='relu'),
            )
        self.decoder_3 = nn.Sequential(
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        input_max, max_indices= torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2)

        return decoder_3
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

# model from 27
# improve by relu
class EDOF_CNN_pack_33(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_33, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 16, 5, 1, activation='relu')
        )
        self.encoder_1 = nn.Sequential(
            PackLayerConv3d(16, 5, activation='relu')
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(16, 32, 3, 1, activation='relu'),
            PackLayerConv3d(32, 3, activation='relu')
        )
        self.encoder_3 = nn.Sequential(
            Conv2D(32, 64, 3, 1, activation='relu'),
            PackLayerConv3d(64, 3, activation='relu')
        )
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(64, 64, 3, activation='relu'),
            Conv2D(64, 32, 3, 1, activation='relu')
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3, activation='relu'),
            Conv2D(32, 16, 3, 1, activation='relu'),
            )
        self.decoder_3 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3, activation='relu'),
            )
        self.decoder_4 = nn.Sequential(
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        encoder_3 = [self.encoder_3(X) for X in encoder_2]
        input_max, max_indices= torch.min(torch.stack(encoder_3),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip3, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2) + skip3
        decoder_4 = self.decoder_4(decoder_3)

        return decoder_4
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

# model from 33
# improve by redundant 32 channel
class EDOF_CNN_pack_34(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_34, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 16, 5, 1, activation='relu')
        )
        self.encoder_1 = nn.Sequential(
            PackLayerConv3d(16, 5, activation='relu')
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(16, 32, 3, 1, activation='relu'),
            PackLayerConv3d(32, 3, activation='relu')
        )
        self.encoder_3 = nn.Sequential(
            Conv2D(32, 32, 3, 1, activation='relu'),
            PackLayerConv3d(32, 3, activation='relu')
        )
        self.encoder_4 = nn.Sequential(
            Conv2D(32, 64, 3, 1, activation='relu'),
            PackLayerConv3d(64, 3, activation='relu')
        )
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(64, 64, 3, activation='relu'),
            Conv2D(64, 32, 3, 1, activation='relu')
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3, activation='relu'),
            Conv2D(32, 32, 3, 1, activation='relu'),
            )
        self.decoder_3 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3, activation='relu'),
            Conv2D(32, 16, 3, 1, activation='relu'),
            )
        self.decoder_4 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3, activation='relu'),
            )
        self.decoder_5 = nn.Sequential(
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        encoder_3 = [self.encoder_3(X) for X in encoder_2]
        encoder_4 = [self.encoder_4(X) for X in encoder_3]
        input_max, max_indices= torch.min(torch.stack(encoder_4),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_3),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip3, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip4, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2) + skip3
        decoder_4 = self.decoder_4(decoder_3) + skip4
        decoder_5 = self.decoder_5(decoder_4)

        return decoder_5
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

# model from 33
# improve by convolution follow pack unpack
class EDOF_CNN_pack_35(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_35, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 16, 5, 1, activation='relu')
        )
        self.encoder_1 = nn.Sequential(
            Conv2D(16, 16, 3, 1, activation='relu'),
            PackLayerConv3d(16, 5, activation='relu')
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(16, 32, 3, 1, activation='relu'),
            PackLayerConv3d(32, 3, activation='relu')
        )
        self.encoder_3 = nn.Sequential(
            Conv2D(32, 64, 3, 1, activation='relu'),
            PackLayerConv3d(64, 3, activation='relu')
        )
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(64, 64, 3, activation='relu'),
            Conv2D(64, 32, 3, 1, activation='relu')
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3, activation='relu'),
            Conv2D(32, 16, 3, 1, activation='relu'),
            )
        self.decoder_3 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3, activation='relu'),
            Conv2D(16, 16, 3, 1, activation='relu')
            )
        self.decoder_4 = nn.Sequential(
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        encoder_3 = [self.encoder_3(X) for X in encoder_2]
        input_max, max_indices= torch.min(torch.stack(encoder_3),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip3, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2) + skip3
        decoder_4 = self.decoder_4(decoder_3)

        return decoder_4
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

# model from 34
# improve by redundant 32 channel
# original 22
# remove relu
class EDOF_CNN_pack_36(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_36, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 16, 5, 1)
        )
        self.encoder_1 = nn.Sequential(
            PackLayerConv3d(16, 5)
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
        )
        self.encoder_3 = nn.Sequential(
            Conv2D(32, 32, 3, 1),
            PackLayerConv3d(32, 3)
        )
        self.encoder_4 = nn.Sequential(
            Conv2D(32, 64, 3, 1),
            PackLayerConv3d(64, 3)
        )
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(64, 64, 3,),
            Conv2D(64, 32, 3, 1)
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 32, 3, 1),
            )
        self.decoder_3 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1),
            )
        self.decoder_4 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3),
            )
        self.decoder_5 = nn.Sequential(
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        encoder_3 = [self.encoder_3(X) for X in encoder_2]
        encoder_4 = [self.encoder_4(X) for X in encoder_3]
        input_max, max_indices= torch.min(torch.stack(encoder_4),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_3),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip3, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip4, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2) + skip3
        decoder_4 = self.decoder_4(decoder_3) + skip4
        decoder_5 = self.decoder_5(decoder_4)

        return decoder_5
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

# model from 33
# improve by convolution follow pack unpack
# original model 27
# remove relu
class EDOF_CNN_pack_37(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_37, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 16, 5, 1)
        )
        self.encoder_1 = nn.Sequential(
            Conv2D(16, 16, 3, 1),
            PackLayerConv3d(16, 5)
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
        )
        self.encoder_3 = nn.Sequential(
            Conv2D(32, 64, 3, 1),
            PackLayerConv3d(64, 3)
        )
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(64, 64, 3),
            Conv2D(64, 32, 3, 1)
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 16, 3, 1),
            )
        self.decoder_3 = nn.Sequential( 
            UnpackLayerConv3d(16, 16, 3),
            Conv2D(16, 16, 3, 1)
            )
        self.decoder_4 = nn.Sequential(
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        encoder_3 = [self.encoder_3(X) for X in encoder_2]
        input_max, max_indices= torch.min(torch.stack(encoder_3),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip3, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2) + skip3
        decoder_4 = self.decoder_4(decoder_3)

        return decoder_4
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

# model from 37
# model from 33
# improve by convolution follow pack unpack
# original model 27
# remove relu
# channel concat
class EDOF_CNN_pack_38(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_38, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 16, 5, 1)
        )
        self.encoder_1 = nn.Sequential(
            Conv2D(16, 16, 3, 1),
            PackLayerConv3d(16, 5)
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
        )
        self.encoder_3 = nn.Sequential(
            Conv2D(32, 64, 3, 1),
            PackLayerConv3d(64, 3)
        )
        self.unpack_1 = UnpackLayerConv3d(64, 64, 3)
        self.iconv_1 = Conv2D(64 + 32, 64, 3, 1)

        self.unpack_2 = UnpackLayerConv3d(64, 32, 3)
        self.iconv_2 = Conv2D(64, 32, 3, 1)

        self.unpack_3 = UnpackLayerConv3d(32, 16, 3)
        self.iconv_3 = Conv2D(32, 16, 3, 1)

        self.decoder_4 = nn.Sequential(
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        encoder_3 = [self.encoder_3(X) for X in encoder_2]

        input_max, max_indices= torch.min(torch.stack(encoder_3),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip3, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)

        unpack_1 = self.unpack_1(input_max)
        concat_1 = torch.cat((unpack_1, skip1), 1)
        iconv_1 = self.iconv_1(concat_1)

        unpack_2 = self.unpack_2(iconv_1)
        concat_2 = torch.cat((unpack_2,skip2),1)
        iconv_2 = self.iconv_2(concat_2)

        unpack_3 = self.unpack_3(iconv_2)
        concat_3 = torch.cat((unpack_3,skip3),1)
        iconv_3 = self.iconv_3(concat_3)

        decoder_4 = self.decoder_4(iconv_3)

        return decoder_4
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

# model from 38
# model from 37
# model from 33
# improve by convolution follow pack unpack
# original model 27
# remove relu
# channel addition packnet style
class EDOF_CNN_pack_38(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_38, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 16, 5, 1)
        )
        self.encoder_1 = nn.Sequential(
            Conv2D(16, 16, 3, 1),
            PackLayerConv3d(16, 5)
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3)
        )
        self.encoder_3 = nn.Sequential(
            Conv2D(32, 64, 3, 1),
            PackLayerConv3d(64, 3)
        )
        self.unpack_1 = UnpackLayerConv3d(64, 32, 3)
        self.iconv_1 = Conv2D(32, 64, 3, 1)

        self.unpack_2 = UnpackLayerConv3d(64, 16, 3)
        self.iconv_2 = Conv2D(16, 32, 3, 1)

        self.unpack_3 = UnpackLayerConv3d(32, 16, 3)
        self.iconv_3 = Conv2D(16, 16, 3, 1)

        self.decoder_4 = nn.Sequential(
            ConvLayer(16, 1, 3, 1, activation='relu'),
            ConvLayer(1, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        encoder_3 = [self.encoder_3(X) for X in encoder_2]

        input_max, max_indices= torch.min(torch.stack(encoder_3),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip3, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)

        unpack_1 = self.unpack_1(input_max)
        concat_1 = unpack_1 + skip1
        iconv_1 = self.iconv_1(concat_1)

        unpack_2 = self.unpack_2(iconv_1)
        concat_2 = unpack_2 + skip2
        iconv_2 = self.iconv_2(concat_2)

        unpack_3 = self.unpack_3(iconv_2)
        concat_3 = unpack_3 + skip3
        iconv_3 = self.iconv_3(concat_3)

        decoder_4 = self.decoder_4(iconv_3)

        return decoder_4
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

# model from 22 
# add initial layer size from 16 to 32
# out of memory
class EDOF_CNN_pack_39(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_39, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 32, 3, 1)
        )
        self.encoder_1 = nn.Sequential(
            PackLayerConv3d(32, 3)
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(32, 64, 3, 1),
            PackLayerConv3d(64, 3)
        )
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(64, 64, 3),
            Conv2D(64, 32, 3, 1)
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 32, 3, 1),
            )
        self.decoder_3 = nn.Sequential(
            ConvLayer(32, 16, 3, 1, activation='relu'),
            ConvLayer(16, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        input_max, max_indices= torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2)

        return decoder_3
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)
    
# model from 22 
# add initial layer size from 16 to 32
# model from 39
# reaarrange packnet layer style
# out of memory convert ti batchsize 4
class EDOF_CNN_pack_40(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_40, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 32, 3, 1)
        )
        self.encoder_1 = nn.Sequential(
            Conv2D(32, 32, 3, 1),
            PackLayerConv3d(32, 3),
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(32, 64, 3, 1),
            PackLayerConv3d(64, 3)
        )
        self.decoder_1 = nn.Sequential( 
            UnpackLayerConv3d(64, 64, 3),
            Conv2D(64, 32, 3, 1)
            )
        self.decoder_2 = nn.Sequential( 
            UnpackLayerConv3d(32, 32, 3),
            Conv2D(32, 32, 3, 1),
            )
        self.decoder_3 = nn.Sequential(
            ConvLayer(32, 16, 3, 1, activation='relu'),
            ConvLayer(16, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        input_max, max_indices= torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)
        decoder_1 = self.decoder_1(input_max) + skip1
        decoder_2 = self.decoder_2(decoder_1) + skip2
        decoder_3 = self.decoder_3(decoder_2)

        return decoder_3
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

# model from 22 
# add initial layer size from 16 to 32
# model from 39
# reaarrange packnet layer style
# model from 40
# packnet addition
# out of memory convert to batchsize 4
class EDOF_CNN_pack_41(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_41, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 32, 3, 1)
        )
        self.encoder_1 = nn.Sequential(
            Conv2D(32, 32, 3, 1),
            PackLayerConv3d(32, 3),
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(32, 64, 3, 1),
            PackLayerConv3d(64, 3)
        )
        self.unpack_1 = UnpackLayerConv3d(64, 32, 3)
        self.iconv_1 = Conv2D(32, 64, 3, 1)

        self.unpack_2 = UnpackLayerConv3d(64, 32, 3)
        self.iconv_2 = Conv2D(32, 32, 3, 1)
 
        self.decoder_3 = nn.Sequential(
            ConvLayer(32, 16, 3, 1, activation='relu'),
            ConvLayer(16, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        input_max, max_indices= torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)

        unpack_1 = self.unpack_1(input_max)
        concat_1 = unpack_1 + skip1
        iconv_1 = self.iconv_1(concat_1)

        unpack_2 = self.unpack_2(iconv_1)
        concat_2 = unpack_2 + skip2
        iconv_2 = self.iconv_2(concat_2)

        decoder_3 = self.decoder_3(iconv_2)

        return decoder_3
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

# model from 22 
# add initial layer size from 16 to 32
# model from 39
# reaarrange packnet layer style
# model from 40
# packnet addition
# model from 41
# addition with updown decoder
# out of memory convert to batchsize 4
class EDOF_CNN_pack_42(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_42, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 16, 3, 1)
        )
        self.encoder_1 = nn.Sequential(
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3),
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(32, 64, 3, 1),
            PackLayerConv3d(64, 3)
        )
        self.unpack_1 = UnpackLayerConv3d(64, 32, 3)
        self.iconv_1 = Conv2D(32, 64, 3, 1)

        self.unpack_2 = UnpackLayerConv3d(64, 16, 3)
        self.iconv_2 = Conv2D(16, 32, 3, 1)
 
        self.decoder_3 = nn.Sequential(
            ConvLayer(32, 16, 3, 1, activation='relu'),
            ConvLayer(16, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        input_max, max_indices= torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)

        unpack_1 = self.unpack_1(input_max)
        concat_1 = unpack_1 + skip1
        iconv_1 = self.iconv_1(concat_1)

        unpack_2 = self.unpack_2(iconv_1)
        concat_2 = unpack_2 + skip2
        iconv_2 = self.iconv_2(concat_2)

        decoder_3 = self.decoder_3(iconv_2)

        return decoder_3
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

# Best performance
# model from 22 
# add initial layer size from 16 to 32
# model from 39
# reaarrange packnet layer style
# model from 40
# model from 41
# packnet concat
# out of memory convert to batchsize 4
class EDOF_CNN_pack_43(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_43, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 32, 3, 1)
        )
        self.encoder_1 = nn.Sequential(
            Conv2D(32, 32, 3, 1),
            PackLayerConv3d(32, 3),
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(32, 64, 3, 1),
            PackLayerConv3d(64, 3)
        )
        self.unpack_1 = UnpackLayerConv3d(64, 64, 3)
        self.iconv_1 = Conv2D(64 + 32, 64, 3, 1)

        self.unpack_2 = UnpackLayerConv3d(64, 32, 3)
        self.iconv_2 = Conv2D(64, 32, 3, 1)
 
        self.decoder_3 = nn.Sequential(
            ConvLayer(32, 16, 3, 1, activation='relu'),
            ConvLayer(16, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        input_max, max_indices= torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)

        unpack_1 = self.unpack_1(input_max)
        concat_1 = torch.cat((unpack_1, skip1), 1)
        iconv_1 = self.iconv_1(concat_1)

        unpack_2 = self.unpack_2(iconv_1)
        concat_2 = torch.cat((unpack_2, skip2), 1)
        iconv_2 = self.iconv_2(concat_2)

        decoder_3 = self.decoder_3(iconv_2)

        return decoder_3
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

# model from 43
# set intial to 16
class EDOF_CNN_pack_44(nn.Module):    
    def __init__(self):        
        super(EDOF_CNN_pack_44, self).__init__()

        self.encoder_0 = nn.Sequential(
            Conv2D(1, 16, 3, 1)
        )
        self.encoder_1 = nn.Sequential(
            Conv2D(16, 32, 3, 1),
            PackLayerConv3d(32, 3),
        )
        self.encoder_2 = nn.Sequential(
            Conv2D(32, 64, 3, 1),
            PackLayerConv3d(64, 3)
        )
        self.unpack_1 = UnpackLayerConv3d(64, 64, 3)
        self.iconv_1 = Conv2D(64 + 32, 64, 3, 1)

        self.unpack_2 = UnpackLayerConv3d(64, 32, 3)
        self.iconv_2 = Conv2D(32 + 16, 32, 3, 1)
 
        self.decoder_3 = nn.Sequential(
            ConvLayer(32, 16, 3, 1, activation='relu'),
            ConvLayer(16, 1, 1, 1, activation='linear')
        )
        
    def forward(self, XX):
        encoder_0 = [self.encoder_0(X) for X in XX]
        encoder_1 = [self.encoder_1(X) for X in encoder_0]
        encoder_2 = [self.encoder_2(X) for X in encoder_1]
        input_max, max_indices= torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
        skip1, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
        skip2, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)

        unpack_1 = self.unpack_1(input_max)
        concat_1 = torch.cat((unpack_1, skip1), 1)
        iconv_1 = self.iconv_1(concat_1)

        unpack_2 = self.unpack_2(iconv_1)
        concat_2 = torch.cat((unpack_2, skip2), 1)
        iconv_2 = self.iconv_2(concat_2)

        decoder_3 = self.decoder_3(iconv_2)

        return decoder_3
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)

class EDOF_CNN_pack_ensemble(nn.Module):    
    def __init__(self, modelR, modelG, modelB):        
        super(EDOF_CNN_pack_ensemble, self).__init__()
        self.modelR = modelR
        self.modelG = modelG
        self.modelB = modelB
        
    def forward(self, XX):
        XX_red = [X[:,[0],:,:] for X in XX]
        XX_green = [X[:,[1],:,:] for X in XX]
        XX_blue = [X[:,[2],:,:] for X in XX]
        def process(model,XX):
            encoder_0 = [model.encoder_0(X) for X in XX]
            encoder_1 = [model.encoder_1(X) for X in encoder_0]
            encoder_2 = [model.encoder_2(X) for X in encoder_1]
            input_max, max_indices= torch.min(torch.stack(encoder_2),dim=0,keepdim=False)
            skip1, _ = torch.min(torch.stack(encoder_1),dim=0,keepdim=False)
            skip2, _ = torch.min(torch.stack(encoder_0),dim=0,keepdim=False)

            unpack_1 = model.unpack_1(input_max)
            concat_1 = torch.cat((unpack_1, skip1), 1)
            iconv_1 = model.iconv_1(concat_1)

            unpack_2 = model.unpack_2(iconv_1)
            concat_2 = torch.cat((unpack_2, skip2), 1)
            iconv_2 = model.iconv_2(concat_2)

            decoder_3 = model.decoder_3(iconv_2)

            return decoder_3
        decoded_red = process(self.modelR,XX_red)
        decoded_green = process(self.modelG,XX_green)
        decoded_blue = process(self.modelB,XX_blue)
        stacked = torch.cat((decoded_red,decoded_green,decoded_blue),1)
        return stacked
    
    def loss(self, Yhat, Y):
        return mse(Yhat, Y)



#print parameters of models
model_edofmax=EDOF_CNN_max()
model_edof3d =EDOF_CNN_3D(5)
model_edoffast=EDOF_CNN_fast()
model_edofpair=EDOF_CNN_pairwise()
model_edofpack=EDOF_CNN_pack()
model_edofpack_02=EDOF_CNN_pack_02()
model_edofpack_03=EDOF_CNN_pack_03()
model_edofpack_04=EDOF_CNN_pack_04()
model_edofpack_05=EDOF_CNN_pack_05()
model_edofpack_06=EDOF_CNN_pack_06()
model_edofpack_07=EDOF_CNN_pack_07()
model_edofpack_08=EDOF_CNN_pack_08()
model_edofpack_09=EDOF_CNN_pack_09()
model_edofpack_10=EDOF_CNN_pack_10()
model_edofpack_11=EDOF_CNN_pack_11()
model_edofpack_12=EDOF_CNN_pack_12()
model_edofpack_13=EDOF_CNN_pack_13()
model_edofpack_14=EDOF_CNN_pack_14()
model_edofpack_15=EDOF_CNN_pack_15()
model_edofpack_16=EDOF_CNN_pack_16()
model_edofpack_17=EDOF_CNN_pack_17()
model_edofpack_18=EDOF_CNN_pack_18()
model_edofpack_19=EDOF_CNN_pack_19()
model_edofpack_20=EDOF_CNN_pack_20()
model_edofpack_21=EDOF_CNN_pack_21()
model_edofpack_22=EDOF_CNN_pack_22()
model_edofpack_23=EDOF_CNN_pack_23()
model_edofpack_24=EDOF_CNN_pack_24()
model_edofpack_25=EDOF_CNN_pack_25()
model_edofpack_26=EDOF_CNN_pack_26()
model_edofpack_27=EDOF_CNN_pack_27()
model_edofpack_28=EDOF_CNN_pack_28()
model_edofpack_29=EDOF_CNN_pack_29()
model_edofpack_30=EDOF_CNN_pack_30()
model_edofpack_31=EDOF_CNN_pack_31()
model_edofpack_32=EDOF_CNN_pack_32()
model_edofpack_33=EDOF_CNN_pack_33()
model_edofpack_34=EDOF_CNN_pack_34()
model_edofpack_35=EDOF_CNN_pack_35()
model_edofpack_36=EDOF_CNN_pack_36()
model_edofpack_37=EDOF_CNN_pack_37()
model_edofpack_38=EDOF_CNN_pack_38()
model_edofpack_39=EDOF_CNN_pack_39()
model_edofpack_40=EDOF_CNN_pack_40()
model_edofpack_41=EDOF_CNN_pack_41()
model_edofpack_42=EDOF_CNN_pack_42()
model_edofpack_43=EDOF_CNN_pack_43()
model_edofpack_44=EDOF_CNN_pack_44()

# model_edofmod =EDOF_CNN_modified()
print("EDOF_CNN_max: ",sum(p.numel() for p in model_edofmax.encoder.parameters())*6+sum(p.numel() for p in model_edofmax.parameters()))
print("EDOF_CNN_3D: ",sum(p.numel() for p in model_edof3d.parameters()))
print("EDOF_CNN_fast: ",sum(p.numel() for p in model_edoffast.encoder.parameters())*6+sum(p.numel() for p in model_edoffast.parameters()))
print("EDOF_CNN_pairwise: ",sum(p.numel() for p in model_edofpair.encoder.parameters())*6+sum(p.numel() for p in model_edofpair.parameters()))
print("EDOF_CNN_pack_02: ",sum(p.numel() for p in model_edofpack_02.encoder.parameters())*6+sum(p.numel() for p in model_edofpack_02.parameters()))
print("EDOF_CNN_pack_03: ",sum(p.numel() for p in model_edofpack_03.encoder.parameters())*6+sum(p.numel() for p in model_edofpack_03.parameters()))
print("EDOF_CNN_pack_04: ",sum(p.numel() for p in model_edofpack_04.encoder.parameters())*6+sum(p.numel() for p in model_edofpack_04.parameters()))
print("EDOF_CNN_pack_05: ",sum(p.numel() for p in model_edofpack_05.encoder.parameters())*6+sum(p.numel() for p in model_edofpack_05.parameters()))
print("EDOF_CNN_pack_06: ",sum(p.numel() for p in model_edofpack_06.encoder.parameters())*6+sum(p.numel() for p in model_edofpack_06.parameters()))
print("EDOF_CNN_pack_09: ",sum(p.numel() for p in model_edofpack_09.encoder.parameters())*6+sum(p.numel() for p in model_edofpack_09.parameters()))
print("EDOF_CNN_pack_11: ",sum(p.numel() for p in model_edofpack_11.encoder.parameters())*6+sum(p.numel() for p in model_edofpack_11.parameters()))
# print("PackNet01: ",sum(p.numel() for p in model_packnet.encoder.parameters())*6+sum(p.numel() for p in model_edofpack.parameters()))
