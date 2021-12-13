import torch
import torch.nn as nn
import torchvision
import sys
import numpy as np
import skimage.color as sc
from PIL import Image
import PIL
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.nn import functional as F


def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img


def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size.
        Args:
            path: path to image
            imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)
    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0] != -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)
    img_np = pil_to_np(img)
    return img, img_np


def pil_to_np(img_PIL):
    """Converts image in PIL format to np.array.
    From W x H x C [0...255] to C x W x H [0..1]
    """
    ar = np.array(img_PIL)
    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]
    return ar.astype(np.float32) / 255.


def get_image_rgb2ycbcr(path, imsize=-1):
    img, _ = get_image(path, imsize)
    img = np.array(img)
    if img.ndim == 3:
        img = sc.rgb2ycbcr(img)
    img_ycbcr_np = pil_to_np(img)
    return img_ycbcr_np


def np_to_torch(img_np):
    ''' Converts image in numpy.array to torch.Tensor.
        From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]


def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
        initialized in a specific way.
        Args:
            input_depth: number of channels in the tensor
            method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
            spatial_size: spatial size of the tensor to initialize
            noise_type: 'u' for uniform; 'n' for normal
            var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[-2], spatial_size[-1]]
        # print('get_noise:shape', shape)
        net_input = torch.zeros(shape)
        fill_noise(net_input, noise_type)
        net_input *= var
        # print('get_noise:net_input.size()', net_input.size())
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False
    return net_input


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    torch.manual_seed(0)
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False


def imtransform(inputs, tfm_matrix):
    grid = F.affine_grid(tfm_matrix.unsqueeze(0), inputs.size())
    outputs = F.grid_sample(inputs, grid, mode='bilinear')
    return outputs


def Dux(U):
    # % Forward finite difference operator(x-axis)
    # discrete gradient operators
    # print('Dxu: size of U:', U.size())
    # end_col_diff = U[:,:,:,0] - U[:,:,:,-1]
    end_col_diff1 = U[:, 0] - U[:, -1]
    # end_col_diff = end_col_diff.unsqueeze(axis=3)
    end_col_diff = end_col_diff1.unsqueeze(axis=1)
    # print('Dux end_col_diff.size()', end_col_diff.size())
    # print('Dux torch.diff(U,1,1).size()', torch.diff(U,1,1).size())
    # dux = torch.cat((torch.diff(U, 1, 3), end_col_diff), axis=3)
    dux = torch.cat((torch.diff(U, 1, 1), end_col_diff), axis=1)
    return dux


def Duy(U):
    # % Forward finite difference operator(x-axis)
    # discrete gradient operators
    # end_row_diff = torch.unsqueeze(U[:,:,0,:] - U[:,:,-1,:], axis=0)
    end_row_diff = U[0, :] - U[-1, :]
    end_row_diff = end_row_diff.unsqueeze(axis=0)
    duy = torch.cat((torch.diff(U, 1, 0), end_row_diff), axis=0)
    return duy


def SVD_shrinkage(I, tau):
    U, S, V = torch.svd(I)
    mask = torch.where(S > tau, 1, 0)
    S = mask * (S - tau)
    # print('SVD_shrinkage: U.size(), S.size()', U.size(), S.size())
    S = torch.diag(S)
    temp = torch.matmul(U, S)
    I = torch.matmul(temp, torch.transpose(V, 0, 1))
    return I


def soft_L1_shrinkage(dx, thread):
    s = abs(dx)
    # scalar_zero = torch.tensor(0)
    dx = torch.max(s-thread, torch.zeros_like(dx)) * torch.sign(dx)
    return dx


def zero_pad(image, shape, position='corner'):
    # shape = torch.IntTensor(shape)
    # imshape = torch.IntTensor(image.shape)
    # print(shape)
    # print(imshape)
    shape = np.asarray(shape, dtype=int)[-2:]
    imshape = np.asarray(image.shape, dtype=int)[-2:]
    # print("zero_pad imshape", imshape)
    # print("zero_pad shape", shape)

    if np.all(imshape == shape):
        return image
    if np.any(shape < 0):
        raise ValueError("ZERO_PAD: null or negative shape given")
    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")
    pad_img = torch.zeros(tuple(shape), dtype=image.dtype)
    # idx, idy = np.indices(shape)
    idx, idy = torch.meshgrid(torch.arange(0, imshape[0], dtype=torch.long), torch.arange(0, imshape[1], dtype=torch.long))
    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)
    pad_img[idx + offx, idy + offy] = image
    return pad_img


def psf2otf(psf, shape):
    """
       Convert point-spread function to optical transfer function.
       Compute the Fast Fourier Transform (FFT) of the point-spread
       function (PSF) array and creates the optical transfer function (OTF)
       array that is not influenced by the PSF off-centering.
       By default, the OTF array is the same size as the PSF array.
       To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
       post-pads the PSF array (down or to the right) with zeros to match
       dimensions specified in OUTSIZE, then circularly shifts the values of
       the PSF array up (or to the left) until the central pixel reaches (1,1)
       position.
       Notes
       -----
       Adapted from MATLAB psf2otf function
    """
    if torch.all(psf == 0):
        return torch.zeros_like(psf)
    inshape = psf.shape
    # Pad the PSF to outsize
    psf_pad = zero_pad(psf, shape, position='corner').type(psf.dtype)
    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    # print('psf2otf torch.roll axis_size', inshape)
    for axis, axis_size in enumerate(inshape):
        psf_pad = torch.roll(psf_pad, -int(axis_size / 2), dims=axis)
    # Compute the OTF
    otf = torch.fft.fft2(psf_pad)
    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = torch.sum(torch.tensor(psf_pad.shape).type_as(psf_pad) * torch.log2(torch.tensor(psf.shape).type_as(psf_pad)))
    # 该如何选取还未确定
    # otf[torch.abs(otf.imag) < n_ops * 2.22e-16].imag = torch.tensor(0).type_as(psf_pad)
    # otf[torch.abs(otf.imag) < n_ops * 2.22e-16] = otf[torch.abs(otf.imag) < n_ops * 2.22e-16].real
    if torch.all(torch.abs(otf.imag) < n_ops * 2.22e-16):
        otf = otf.real
    # print(otf)
    return otf


def getC(sizeF, dtype):
    # compute fixed quantities
    #diff_kernelX = torch.unsqueeze(torch.tensor([1, -1]), 1).type(torch.float32)
    diff_kernelX = torch.unsqueeze(torch.tensor([1, -1]), 0).type(dtype)
    #diff_kernelY = torch.unsqueeze(torch.tensor([1, -1]), 0).type(torch.float32)
    diff_kernelY = torch.unsqueeze(torch.tensor([1, -1]), 1).type(dtype)

    # discrete fourier transform of kernel (equal to eigenvalues due to block circular prop.)
    otfDx = psf2otf(diff_kernelX, sizeF)
    otfDy = psf2otf(diff_kernelY, sizeF)
    conjoDx = torch.conj(otfDx)
    conjoDy = torch.conj(otfDy)
    Denom1 = torch.pow(torch.abs(otfDx), 2)
    Denom2 = torch.pow(torch.abs(otfDy), 2)
    return (otfDx, otfDy, conjoDx, conjoDy, Denom1, Denom2)


def boundary_padding(image, pad, mode):
    # print("bondary_padding", pad, mode)
    return F.pad(image, pad, mode=mode)


def crop_image(img, new_size):
    print(img.size())
    diff2 = (img.size(2) - new_size[2]) // 2
    diff3 = (img.size(3) - new_size[3]) // 2
    # print('crop_image:diff2, diff3', diff2, diff3)
    # print('crop_image: img size:', img.size())
    # print('crop_image: new_size:', new_size)
    cropped_img = img[:, :, diff2 : diff2 + new_size[2], diff3 : diff3 + new_size[3]]
    # print('crop_image: cropped_img size:', cropped_img.size())
    # print(cropped_img.squeeze())
    return cropped_img


def trans_crop(O, a_m, new_size):
    O_a = imtransform(O, a_m)
    O_a_c = crop_image(O_a, new_size)
    return O_a_c


def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.
    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def nuclear_norm(R):
    '''compute the nuclear norm of matrix R'''
    U, S, V = torch.svd(R)
    return torch.sum(S)













