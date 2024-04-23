import numpy as np
from time import time
from torch import optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import torch
import dataset, models
from tqdm import tqdm
from PIL import Image
import cv2

pth_path = 'dataset-fraunhofer_elastic_only-image_size-512-method-EDOF_CNN_RGB-Z-5-fold-0-epochs-100-batchsize-1-lr-0.001-cudan-0-image_channels-rgb-automate-1-augmentation-0.pth'
model = models.EDOF_CNN_RGB()
model.load_state_dict(torch.load(pth_path))
print(model.eval())

img_path = r"/home/van/research/EDoF_FhP/aligned/Focus_smear01/x3_y21/4968.jpg"
image = cv2.imread(img_path)
image = np.array(image)