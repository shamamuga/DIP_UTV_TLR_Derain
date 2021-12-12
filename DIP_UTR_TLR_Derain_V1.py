from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from models import fcn
from models import unet
import torch
import torch.optim
import warnings
import glob
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from utils.commom_utils import *
from models.unet import UNet


parser = argparse.ArgumentParser()
parser.add_argument('--num_iter', type=int, default=5000, help='number of epochs of training')
parser.add_argument('--img_size', type=int, default=[256, 256], help='size of each image dimension')
parser.add_argument('--data_path', type=str, default='datasets/', help='path to rainny image')
parser.add_argument('--save_path', type=str, default='results/', help='path to save results')
parser.add_argument('--save_frequency', type=int, default=100, help='lfrequency to save results')
opt = parser.parse_args()


# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark =True
# dtype = torch.cuda.FloatTensor
dtype = torch.float32
warnings.filterwarnings("ignore")
file_source = glob.glob(os.path.join(opt.data_path, '*.png'))
file_source.sort()
save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)

# start # image
for f in file_source:
    INPUT = 'noise'
    LR = 0.01
    num_iter = opt.num_iter
    reg_noise_std = 0.01
    path_to_image = f
    _, imgs = get_image(path_to_image, -1)  # load image and convert to np.
    y = np_to_torch(imgs).type_as(dtype)
    img_size = imgs.shape
    # ######################################################################
    '''
            x_net:
    '''
    input_depth = 2
    net_input = get_noise(input_depth, INPUT, (opt.img_size[0], opt.img_size[1])).type(dtype)
    net = UNet(input_depth, 1)
    net = net.type(dtype)
    '''
            net_affine:
    '''
    n_k = 200
    net_input_affine = get_noise(n_k, INPUT, (1, 1)).type(dtype)
    net_input_affine.squeeze_()
    net_affine = fcn(n_k, 4)
    net_affine = net_affine.type(dtype)


    # Losses
    mse = torch.nn.MSELoss().type(dtype)

    # optimizer
    optimizer = torch.optim.Adam([{'params': net.parameters()}, {'params': net_affine.parameters(), 'lr':1e-4}], lr=LR)
    scheduler = MultiStepLR(optimizer, milestones=[2000, 3000, 4000], gamma=0.5) # learning rates
    #
    net_input_saved = net_input.detach().clone()
    net_input_affine_saved = net_input_affine.detach().clone()

    ### start DIP_UTR_TLR_Derain_V1
    for step in tqdm(range(num_iter)):
        # input regularization
        net_input = net_input_saved + reg_noise_std * torch.zeros(net_input_saved.shape).type_as(net_input_saved.data).normal_()
        # net_input_affine = net_input_affine_saved + reg_noise_std*torch.zeros(net_input_affine_saved.shape).type_as(net_input_affine_saved.data).normal_()

        # change the learning rate
        scheduler.step(step)
        optimizer.zero_grad()
        # get the network output
        out_x = net(net_input)
        out_a = net_affine(net_input_affine)
        out_a_m = out_a.view(-1, 1, 2, 2)
        # print(out_a_m)
        out_y = imtransform(y, out_a)
        total_loss = mse(out_x, out_y)
        total_loss.backward()
        optimizer.step()
        out_a = net_affine(net_input_affine)
        out_a_m = out_a.view(-1, 1, 2, 2)
        out_y = imtransform(y, out_a)
        dx_out_x = Dux(out_x)
        total_loss1 = mse(out_y - out_x) + mu * mse(dx_out_x, V2)


        '''# start # image
        class Param():
            pass


        param = Param()
        param.mu = opt.mu
        param.lambda1 = opt.lambda1
        param.lambda2 = opt.lambda2
        param.lambda3 = opt.lambda3
        param.iter_num = opt.num_outer__iter
        param.inner_iter_num = opt.num_inner_iter'''




