from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
from skimage.io import imsave
from skimage.io import imshow
import matplotlib.pyplot as plt
import os
import numpy as np
from models import fcn
from models.unet import UNet
import torch
import torch.optim
import glob
from tqdm import tqdm
import warnings
from utils.commom_utils import *
from ADMM_RTL_UTV_DIP import *
import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



parser = argparse.ArgumentParser()
parser.add_argument('--num_outer_iter', type=int, default=1, help='number of epochs of training')
parser.add_argument('--num_inner_iter_cnn', type=int, default=2, help='number of inner ADAM iterations for cnn')
parser.add_argument('--num_inner_iter_affine', type=int, default=2, help='number of inner ADAM iterations for affine')
parser.add_argument('--data_path', type=str, default='datasets/', help='path to rainny image')
parser.add_argument('--img_size', type=int, default=[256, 256], help='size of each image dimension')
parser.add_argument('--save_path', type=str, default='results/', help='path to save results')
parser.add_argument('--save_frequency', type=int, default=100, help='frequency to save results')
parser.add_argument('--mu', type=float, default=0.5, help='augmented lagrange parameter')
parser.add_argument('--lambda1', type=float, default=0.4, help='kernel norm weight')
parser.add_argument('--lambda2', type=float, default=0.4, help='x-axis variation norm')
parser.add_argument('--lambda3', type=float, default=0.4, help='y-axis variation norm')


opt = parser.parse_args()

#dtype = torch.float32
warnings.filterwarnings("ignore")


# file_source = glob.glob(os.path.join(opt.data_path, '*.png'))
file_source = glob.glob(os.path.join(opt.data_path, '*.bmp'))
file_source.sort()
save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)

print('start running')
for f in file_source:
    path_to_image = f
    imgname = os.path.basename(f)
    imgname = os.path.splitext(imgname)[0]
    # _, imgs = get_image(path_to_image, -1)  # load image and convert to np.
    img_ycbcr = get_image_rgb2ycbcr(path_to_image)
    imgs = img_ycbcr[0, :, :]
    imgs = imgs[None, ...]
    #O = np_to_torch(imgs).type(dtype)
    O = np_to_torch(imgs)
    shape = O.size()
    m = shape[-2]
    n = shape[-1]
    opt.img_size = shape
    pad_left = np.int32(np.ceil((np.sqrt(2)*np.sqrt(m*m + n*n) - n)/2))
    pad_right = pad_left
    pad_up = np.int(np.ceil((np.sqrt(2)*np.sqrt(m*m + n*n) - m)/2))
    pad_down = pad_up
    pad = (pad_left, pad_right, pad_up, pad_down)
    O_pad = boundary_padding(O, pad, mode='circular')
    O_pad_np = torch_to_np(O_pad).squeeze()
    save_path = os.path.join(opt.save_path, '%s_x.png' % imgname)
    imsave(save_path, O_pad_np)
    # print('O_pad_np:', O_pad_np)
    # print(save_path)
    rain_free_img, rain_streak, total_loss_list = ADMM_RTL_UTV_DIP1(O_pad, opt)
    print(rain_streak.size())
    out_x_np = torch_to_np(rain_free_img)
    rain_np = rain_streak.detach().cpu().numpy()
    # rain_np = torch_to_np(rain_streak)
    save_path = os.path.join(opt.save_path, '%s_x.png' % imgname)
    out_x_np = out_x_np.squeeze()*255
    rain_np = rain_np*255
    print(np.shape(rain_np))
    # print('out_x_np', out_x_np)
    imshow(out_x_np)
    plt.show()
    imshow(rain_np)
    plt.show()
    imsave(save_path, out_x_np)
    print('total_loss_list', total_loss_list)










