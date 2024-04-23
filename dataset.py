# =============================================================================
# Dataset loader for the training process
# =============================================================================

from torch.utils.data import Dataset
from torchvision import models, transforms
import pickle
import torch
import numpy as np
import cv2
import glob
from PIL import Image, ImageOps
import PIL
import os 
from utils_files.automatic_brightness_and_contrast import automatic_brightness_and_contrast
from augmentfunctions_tf import aug_chromab
import tensorflow as tf
import random
from joblib import dump, load

class Dataset(Dataset):
    def __init__(self, type, transform, dataset, Z, fold, target_transform, common_transform):
        self.X, self.Y = pickle.load(open(f'dataset/data_{dataset}.pickle', 'rb'))[fold][type]
        # self.X, self.Y = load('dataset/data_'+dataset+'.joblib')[fold][type]
        self.transform = transform
        self.target_transform = target_transform
        self.Z = Z
        self.common_transform = common_transform
    def __getitem__(self, i):
        X0 = self.transform(self.X[i][0])
        X1 = self.transform(self.X[i][1])
        X2 = self.transform(self.X[i][2])

     
        if self.Z==5:
            X3 = self.transform(self.X[i][3])
            X4 = self.transform(self.X[i][4])
        elif self.Z==7:
            X3 = self.transform(self.X[i][3])
            X4 = self.transform(self.X[i][4])
            X5 = self.transform(self.X[i][5])
            X6 = self.transform(self.X[i][6])
        elif self.Z==9:
            X3 = self.transform(self.X[i][3])
            X4 = self.transform(self.X[i][4])
            X5 = self.transform(self.X[i][5])
            X6 = self.transform(self.X[i][6])
            X5 = self.transform(self.X[i][5])
            X6 = self.transform(self.X[i][6])
            X7 = self.transform(self.X[i][7])
            X8 = self.transform(self.X[i][8])

        Y = self.target_transform(self.Y[i])
        
        if self.Z==3:
            X_all=[X0, X1, X2]
        elif self.Z==5:
            X_all=[X0, X1, X2, X3, X4]
        elif self.Z==7:
            X_all=[X0, X1, X2, X3, X4, X5, X6]
        elif self.Z==9:
            X_all=[X0, X1, X2, X3, X4, X5, X6, X7, X8]
            
        r=torch.randperm(self.Z)
        
        #return X_all, Y
        randomized_X_all = [X_all[u] for u in r]
        randomized_X_all.append(Y)
        common_transformed = self.common_transform(randomized_X_all)
        return common_transformed[:-1],common_transformed[-1]
        # return [X_all[u] for u in r],Y

    def __len__(self):
        return len(self.X)




class Dataset_folder(Dataset):
    def __init__(self, transform, path, Z, img_size):
        self.path = path
        X_STACKS_per_folder=[]
        for idxx,image_name in enumerate(glob.glob(os.path.join(self.path, "*.jpg"))):
            # im = cv2.imread(image_name)
            # imnew=cv2.resize(im,(img_size,img_size))
            X_STACKS_per_folder.append(image_name)
        
        self.X = np.array(X_STACKS_per_folder)
        self.X = np.expand_dims(self.X, axis=0)
        self.transform = transform
        self.Z = Z
        self.img_size = img_size
        
    def __getitem__(self, i):
        def read_image(image_name):
            im = cv2.imread(image_name)
            imnew=cv2.resize(im,(self.img_size,self.img_size))
            return imnew
        def my_process(my_image):
            my_res = self.transform(my_image)
            return my_res
        X0 = my_process(self.X[i][0])
        X1 = my_process(self.X[i][1])
        X2 = my_process(self.X[i][2])

     
        if self.Z==5:
            X3 = my_process(self.X[i][3])
            X4 = my_process(self.X[i][4])
        elif self.Z==7:
            X3 = my_process(self.X[i][3])
            X4 = my_process(self.X[i][4])
            X5 = my_process(self.X[i][5])
            X6 = my_process(self.X[i][6])
        elif self.Z==9:
            X5 = my_process(self.X[i][5])
            X6 = my_process(self.X[i][6])
            X7 = my_process(self.X[i][7])
            X8 = my_process(self.X[i][8])

        # Y = self.transform(self.Y[i])
        
        if self.Z==3:
            X_all=[X0, X1, X2]
        elif self.Z==5:
            X_all=[X0, X1, X2, X3, X4]
        elif self.Z==7:
            X_all=[X0, X1, X2, X3, X4, X5, X6]
        elif self.Z==9:
            X_all=[X0, X1, X2, X3, X4, X5, X6, X7, X8]
            
        r=torch.randperm(self.Z)
        
        #return X_all, Y
        return [X_all[u] for u in r]

    def __len__(self):
        return len(self.X)
    

class ChromaticAberration():
    """
    Perform Chromatic aberration on the image
    Code was taken from https://github.com/alexacarlson/SensorEffectAugmentation
    """
    def __init__(self, scaleMin, scaleMax, tMin, tMax):
        self.scaleMin = scaleMin
        self.scaleMax = scaleMax
        self.tMin = tMin
        self.tMax = tMax
        
    def __call__(self,image):
        # Hackyway to cooperate with batchsize format
        batch_img = tf.expand_dims(image, axis=0)
        float_img = tf.image.convert_image_dtype(batch_img, tf.float32)
        
        batchsize = 1
        # my_test = np.random.default_rng().uniform(low=0.986,high=1.014, size=(batchsize,1,1,1))
        scale_val = tf.random.uniform((batchsize,1,1,1), minval = self.scaleMin, maxval = self.scaleMax, dtype=tf.float32)
        tx_Rval = tf.random.uniform((batchsize,1,1,1), minval=self.tMin, maxval = self.tMax, dtype=tf.float32)
        ty_Rval = tf.random.uniform((batchsize,1,1,1), minval=self.tMin, maxval = self.tMax, dtype=tf.float32)
        tx_Gval = tf.random.uniform((batchsize,1,1,1), minval=self.tMin, maxval = self.tMax, dtype=tf.float32)  
        ty_Gval = tf.random.uniform((batchsize,1,1,1), minval=self.tMin, maxval = self.tMax, dtype=tf.float32)
        tx_Bval = tf.random.uniform((batchsize,1,1,1), minval=self.tMin, maxval = self.tMax, dtype=tf.float32)  
        ty_Bval = tf.random.uniform((batchsize,1,1,1), minval=self.tMin, maxval = self.tMax, dtype=tf.float32)
        
        crop_h = image.shape[0]
        crop_w = image.shape[1]

        
        augImgBatch = aug_chromab(float_img, crop_h, crop_w, scale_val, tx_Rval, ty_Rval, tx_Gval, ty_Gval, tx_Bval, ty_Bval)
        # # batch img to image
        augImg = np.array(augImgBatch[0])
        augImg = augImg * 255
        augImg = augImg.astype(np.uint8)
        return augImg
    
class MyRotation():
    def __init__(self,stop=0):
        self.stop = stop
    def __call__(self,images):
        rand = torch.rand(1).numpy()[0]
        random_angle = rand * 360
        if self.stop == 1:
            random_angle = 0
        elif self.stop == 2:
            random_angle = random.choice([-30, -15, 0, 15, 30])
        elif self.stop ==3:
            if random.random() > 0.5:
                random_angle = 0
        result = []
        for img in images:
            result.append(transforms.functional.rotate(img,random_angle))
        return result
    
class MyOneChannel():
    def __init__(self,chan):
        self.chan = chan
    def __call__(self,image):
        extract_image = image[:,:,self.chan]
        return extract_image
    
class MyPILOneChannel():
    def __init__(self,chan):
        self.chan = chan
    def __call__(self,images):
        result = []
        for img in images:
            result.append(img.getchannel(self.chan))
        return result
    
class MyToTensor():
    def __init__(self):
        pass
    def __call__(self,images):
        result = []
        for img in images:
            result.append(transforms.functional.to_tensor(img))
        return result
    
class EmptyCommonTransform():
    def __init__(self):
        pass
    def __call__(self,images):
        return images
    
class MyHFlip():
    def __init__(self):
        pass
    def __call__(self,images):
        rand = torch.rand(1).numpy()[0]
        result = []
        if rand > 0.5:
            for img in images:
                result.append(transforms.functional.hflip(img))
            return result
        else: 
            return images
        
class MyJitter():
    def __init__(self, brightness, contrast, saturation, hue):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    def __call__(self,images):
        def func(img, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor):
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = transforms.functional.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img =  transforms.functional.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img =  transforms.functional.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img =  transforms.functional.adjust_hue(img, hue_factor)
            return img
        color_jitter = transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue)
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = transforms.ColorJitter.get_params(color_jitter.brightness, color_jitter.contrast, color_jitter.saturation,color_jitter.hue)
        result = []
        for img in images:
            result.append(func(img,fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor))
        return result


aug_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    # automatic_brightness_and_contrast(clip_hist_perc=0.5),
    transforms.Grayscale(num_output_channels=1),
    # vgg normalization
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transforms_PIL_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
])

aug_transforms_red = transforms.Compose([
    MyOneChannel(0),
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
])

aug_transforms_red_pil = transforms.Compose([
    transforms.ToPILImage(),
    MyPILOneChannel('R'),
    transforms.Resize((512, 512)),
])

aug_transforms_green = transforms.Compose([
    MyOneChannel(1),
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
])

aug_transforms_blue = transforms.Compose([
    MyOneChannel(2),
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
])

aug_transforms_aug = transforms.Compose([
    ChromaticAberration(scaleMin = 0.986, scaleMax = 1.014, tMin = -0.014, tMax = 0.014),
    transforms.ToPILImage(),
    # transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((512, 512),antialias=True),
    # automatic_brightness_and_contrast(clip_hist_perc=0.5),
    transforms.Grayscale(num_output_channels=1),
    # vgg normalization
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

aug_transforms_aug_1 = transforms.Compose([
    ChromaticAberration(scaleMin = 0.998, scaleMax = 1.002, tMin = -0.002, tMax = 0.002),
    transforms.ToPILImage(),
    # transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((512, 512),antialias=True),
    # automatic_brightness_and_contrast(clip_hist_perc=0.5),
    transforms.Grayscale(num_output_channels=1),
    # vgg normalization
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    # automatic_brightness_and_contrast(clip_hist_perc=0.5),
    transforms.Grayscale(num_output_channels=1),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

aug_transforms_rgb = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    # automatic_brightness_and_contrast(clip_hist_perc=0.5),
    # vgg normalization
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

aug_transforms_rgb_aug = transforms.Compose([
    ChromaticAberration(scaleMin = 0.986, scaleMax = 1.014, tMin = -0.014, tMax = 0.014),
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    # automatic_brightness_and_contrast(clip_hist_perc=0.5),
    # vgg normalization
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

aug_transforms_rgb_aug_1 = transforms.Compose([
    ChromaticAberration(scaleMin = 0.998, scaleMax = 1.002, tMin = -0.002, tMax = 0.002),
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    # automatic_brightness_and_contrast(clip_hist_perc=0.5),
    # vgg normalization
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

aug_transforms_rgb_aug_2 = transforms.Compose([
    ChromaticAberration(scaleMin = 1, scaleMax = 1, tMin = 0, tMax = 0),
    # transforms.ToPILImage(),
    # transforms.Resize((512, 512)),
    # automatic_brightness_and_contrast(clip_hist_perc=0.5),
    # transforms.ToTensor(),
    # vgg normalization
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

val_transforms_rgb = transforms.Compose([
    transforms.ToPILImage(),
    #  transforms.Resize((512, 512)),
    # automatic_brightness_and_contrast(clip_hist_perc=0.5),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

common_transform_empty = transforms.Compose([
    EmptyCommonTransform(),
    MyToTensor()
])

common_transform_rotate = transforms.Compose([
    MyRotation(),
    transforms.ToTensor()
])

common_transform_rotate_2 = transforms.Compose([
    MyRotation(stop=1),
    transforms.ToTensor()
])

common_transform_rotate_3 = transforms.Compose([
    MyRotation(stop=2),
    transforms.ToTensor()
])

common_transform_rotate_4 = transforms.Compose([
    MyRotation(stop=3),
    transforms.ToTensor()
])

common_transform_hflip = transforms.Compose([
    MyHFlip(),
    transforms.ToTensor()
])

common_transform_jitter_1 = transforms.Compose([
    MyJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.2),
    transforms.ToTensor()
])

common_transform_jitter_2 = transforms.Compose([
    MyJitter(brightness=0.8,contrast=0.8,saturation=0.8,hue=0.4),
    transforms.ToTensor()
])

common_transform_jitter_2 = transforms.Compose([
    MyJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.05),
    transforms.ToTensor()
])

transform_jitter_2 = [
    MyJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.05)
]

transform_jitter_3 = [
    MyJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1)
]

transforms_red_pil = [
    MyPILOneChannel('R')
]

transforms_green_pil = [
    MyPILOneChannel('G')
]

transforms_blue_pil = [
    MyPILOneChannel('B')
]

transforms_to_tensor = [
    MyToTensor()
]