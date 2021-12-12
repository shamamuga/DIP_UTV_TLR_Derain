import torch
import torch.nn as nn
import numpy as np
from .downsampler import Downsampler
from torch.nn import functional as F


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':
        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False
    stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)
    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)


def soft_L1_shrinkage(dx, thread):
    s = abs(dx)
    # scalar_zero = torch.tensor(0)
    dx = torch.max(s-thread, torch.zeros_like(dx)) * torch.sign(dx)
    return dx


def soft_Nuclear_shrinkage(s, mu):
    mask = torch.where(s > mu, 1, 0)
    s = mask * (s - mu)
    return s


def SVD_shrinkage(I, tau):
    U, S, V = torch.svd(I)
    mask = torch.where(S > tau, 1, 0)
    S = mask * (S - tau)
    temp = torch.matmul(U, S)
    I = torch.matmul(temp, torch.transpose(V, 0, 1))
    return I


def zero_pad(image, shape, position='corner'):
    shape = torch.LongTensor(shape)
    imshape = torch.IntTensor(image.shape)
    # shape = np.asarray(shape, dtype=int)
    # imshape = np.asarray(image.shape, dtype=int)
    if torch.all(imshape == shape):
        return image
    if torch.any(shape < 0):
        raise ValueError("ZERO_PAD: null or negative shape given")
    dshape = shape - imshape
    if torch.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")
    pad_img = torch.zeros(shape, dtype=image.dtype)
    # idx, idy = np.indices(shape)
    idx, idy = torch.meshgrid(torch.arange(0, shape[0], dtype=torch.long), torch.arange(0, shape[1], dtype=torch.long))
    if position == 'center':
        if torch.any(dshape % 2 != 0):
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
    psf_pad = zero_pad(psf, shape, position='corner')
    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
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
    return otf


def Dux(U):
    # % Forward finite difference operator(x-axis)
    # discrete gradient operators
    end_col_diff = torch.unsqueeze(U[:,:,:,0] - U[:,:,:,1], axis=1)
    Dux = torch.cat((torch.diff(U, 1, 1), end_col_diff), axis=1)
    return Dux


def Duy(U):
    # % Forward finite difference operator(y-axis)
    # discrete gradient operators
    end_rol_diff = torch.unsqueeze(U[:, :, 0, :] - U[:, :, -1, :], axis=0)
    Duy = torch.cat((torch.diff(U, 1, 0), end_rol_diff), axis=0)
    return Duy


def imtransform(inputs, tfm_matrix):
    grid = F.affine_grid(tfm_matrix.unsqueeze(0), inputs.size())
    outputs = F.grid_sample(inputs, grid, mode='bilinear')
    return outputs





















