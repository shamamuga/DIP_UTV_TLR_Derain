import os
import numpy as np
import random
from torch.nn import functional as F
import torch
import glob
from skimage.io import imsave
import matplotlib.pyplot as plt
from utils.commom_utils import *
from adam_network_minimize_back import *
from utils.network_optimization import *
from models.fcn import *
#from models.unet import UNet
from models.unet_origin import UNet
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark =True
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.float32


def ADMM_RTL_UTV_DIP1(O, param):
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark =True
    # dtype = torch.cuda.FloatTensor
    dtype = torch.float32
    # initialize
    size = param.img_size
    m, n = size[-2], size[-1]
    new_size = [1, 1, (np.int(np.ceil(np.sqrt(m * m + n * n)))//16+1)*16, (np.int(np.ceil(np.sqrt(m * m + n * n)))//16+1)*16]
    param.new_size = new_size
    h, w = new_size[-2], new_size[-1]
    R = torch.zeros([h, w]).type(dtype)
    Lam1 = torch.zeros([h, w]).type(dtype)
    Lam2 = torch.zeros([h, w]).type(dtype)
    Lam3 = torch.zeros([h, w]).type(dtype)

    (otfDx, otfDy, conjoDx, conjoDy, Denom1, Denom2) = getC(new_size, dtype)
    total_loss_list = []

    # ######################################################################
    '''
            x_net:
    '''
    INPUT = 'noise'
    input_depth = 2
    net_input = get_noise(input_depth, INPUT, (param.new_size[-2], param.new_size[-1])).type(dtype)
    net = UNet(num_input_channels=input_depth, num_output_channels=1, upsample_mode='deconv', need_sigmoid=False)
    net = net.type(dtype)

    '''
                net_affine:
    '''
    n_k = 200
    net_input_affine = get_noise(n_k, INPUT, (1, 1)).type(dtype)
    net_input_affine.squeeze_()
    # print('net_input_affine', net_input_affine)
    # net_affine = fcn(n_k, 4)
    net_affine = fcn(n_k, 1)
    net_affine = net_affine.type(dtype)

    #
    net_input_saved = net_input.detach().clone()
    print('net_input_saved.shape', net_input_saved.shape)
    net_affine_input_saved = net_input_affine.detach().clone()

    for iter in range(param.num_outer_iter):
        # print('ADMM_RTL_UTV_DIP: net_input_saved.size()', net_input_saved.size())
        out_x = net(net_input_saved)
        out_x = out_x.squeeze()
        # print('ADMM_RTL_UTV.py out_x:', out_x)
        out_a = net_affine(net_affine_input_saved)
        #print("out_a:", out_a)
        # out_a_m = out_a.view(2, 2)
        out_a_m = torch.tensor([[torch.cos(out_a), -torch.sin(out_a)], [torch.sin(out_a), torch.cos(out_a)]])
        out_a_m_p = F.pad(input=out_a_m, pad=(0,1,0,0), mode='constant', value=0).type(dtype)
        # print('ADMM_RTL_UTV.py out_a_m:', out_a_m)
        # O_a = imtransform(O, out_a_m)
        m, n = param.img_size[-2], param.img_size[-1]
        O = O.type(dtype)
        O_a = trans_crop(O, out_a_m_p, new_size)
        O_a_np = torch_to_np(O_a).squeeze()
        save_path = 'results/Afternoon Rain_x.png'
        imsave(save_path, O_a_np)
        O_a = O_a.squeeze()

        total_loss = param.lambda1 * torch.norm(R, p='nuc') + param.lambda2 * torch.norm(Dux(out_x), p=1) \
                     + param.lambda3 * torch.norm(Duy(R), p=1)+0.5*torch.norm(O_a-out_x-R, p='fro')
        total_loss_list.append(total_loss)
        print('in step {step} in ADMM_RTL_UTV_DIP1 total_cost:'.format(step=iter), total_loss)
        # V--subproblem
        V1 = SVD_shrinkage(R + Lam1/param.mu, param.lambda1/param.mu)
        V2 = soft_L1_shrinkage(Dux(out_x) + Lam2/param.mu, param.lambda2/param.mu)
        V3 = soft_L1_shrinkage(Duy(R) + Lam3/param.mu, param.lambda3/param.mu)

        # R--subproblem
        Denom = param.mu * Denom2 + (1 + param.mu)
        Fr = (torch.fft.fft2(O_a - out_x + param.mu*V1 - Lam1) + param.mu*conjoDy*torch.fft.fft2(V3 - Lam3/param.mu))/Denom
        R = torch.real(torch.fft.ifft2(Fr))
        # print('ADMM_RTL_UTV_DIP1: R.size()', R.size())
        # print('ADMM_RTL_UTV_DIP1: R', R)

        # ksai and theta subproblem (i.e network parameter optimization ADAM)
        # (out_x, out_a_m) = adam_network_minmize(net, net_affine, net_input_saved, net_affine_input_saved, O, R, V2, Lam2, param.mu, new_size, param.num_inner_iter)
        # ksai subproblem
        target_affine = out_x + R
        target_affine = target_affine.detach()
        O_a = affine_network_optimization(net_affine, net_affine_input_saved, O, target_affine, param.num_inner_iter_affine, new_size)

        # theta subproblem
        target_net1 = V2 - Lam2 / param.mu
        target_net1 = target_net1.detach()
        target_net2 = O_a - R
        target_net2 = target_net2.detach()
        out_x = cnn_network_optimization(net, net_input_saved, target_net1, target_net2, param.num_inner_iter_cnn, param.mu)

        # Lam update
        Lam1 = Lam1 + param.mu*(R - V1)
        Lam2 = Lam2 + param.mu*(Dux(out_x) - V2)
        Lam3 = Lam3 + param.mu*(Duy(R) - V3)

    return out_x, R, total_loss_list









