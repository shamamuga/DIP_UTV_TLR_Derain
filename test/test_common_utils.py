import unittest
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from models import fcn
from utils.commom_utils import *


class TestCommon_utils(unittest.TestCase):
    def setUp(self):
        self.test_image_path = '../Hills.bmp'
        # self.test_image_path = '../Afternoon Rain.bmp'
        self.test_image_path = '../38.jpg'
        self.dtype = torch.float32

    def test_boundary_padding(self):
        dtype = torch.float32
        img_ycbcr = get_image_rgb2ycbcr(self.test_image_path)
        img = img_ycbcr[0, :, :] * 255
        # im = Image.open(self.test_image_path)
        # implot = plt.imshow(img)
        # img = Image.fromarray(img)
        # plt.show()
        # img.show()
        img = img[None, ...]
        O = np_to_torch(img).type(dtype)
        pad = (100, 100, 100, 100)
        O_pad = boundary_padding(O, pad, mode='circular')
        O_pad_np = torch_to_np(O_pad).squeeze()
        img = Image.fromarray(O_pad_np)
        img.show()

    def test_imtransform_Euclid(self):
        dtype = torch.float64
        theta = np.pi / 13
        # theta = np.pi * 2
        theta = -np.pi/5
        rotation_matrix = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        affine_matrix_a = F.pad(input=rotation_matrix, pad=(0, 1, 0, 0), mode='constant', value=0)
        img_ycbcr = get_image_rgb2ycbcr(self.test_image_path)
        img = img_ycbcr[0, :, :]
        img = img[None, ...]
        O = np_to_torch(img).type(dtype)
        O_a = imtransform(O, affine_matrix_a)
        O_a_np = torch_to_np(O_a).squeeze() * 255
        img = Image.fromarray(O_a_np)
        img.show()

    def test_imtransform_affine(self):
        dtype = torch.float64
        affine_matrix = torch.tensor([[1.0, -0.25], [0.0, 1.0]]).type(dtype)
        affine_matrix_a = F.pad(input=affine_matrix, pad=(0, 1, 0, 0), mode='constant', value=0)
        img_ycbcr = get_image_rgb2ycbcr(self.test_image_path)
        img = img_ycbcr[0, :, :]
        img = img[None, ...]
        O = np_to_torch(img).type(dtype)
        O_a = imtransform(O, affine_matrix_a)
        O_a_np = torch_to_np(O_a).squeeze() * 255
        img = Image.fromarray(O_a_np)
        img.show()

    def test_Dux(self):
        dtype = torch.float32
        U = torch.tensor([[1,3,2,5],[3,5,7,1], [5,0,1,3]]).type(dtype)
        print(Dux(U))
        dux = Dux(U)
        dux = dux.type(dtype)
        print(type(dux))
        result = torch.tensor([[2, -1, 3, -4], [2, 2, -6, 2], [-5, 1, 2, 2]]).type(dtype)
        print(dux - result)
        # self.assertEquals(dux, result)

    def test_Duy(self):
        U = torch.tensor([[1, 3, 2, 5], [3, 5, 7, 1], [5, 0, 1, 3]])
        duy = Duy(U)
        result = torch.tensor([[2, 2, 5, -4], [2, -5, -6, 2], [-4, 3, 1, 2]])
        print(duy)

    def test_SVD_shrinkage(self):
        A = torch.tensor([[4, 3, 6, 3], [6, 4, 3, 2], [6, 5, 7, 6]]).type(self.dtype)
        I = SVD_shrinkage(A, 2)
        print(I)
        R = torch.tensor([[3.7958, 2.9768, 4.3850, 3.1383], [4.3084, 3.1121, 3.3764, 2.4283],
                          [5.6862, 4.4119, 6.2841, 4.4996]])
        I = SVD_shrinkage(A, 4)
        R = torch.tensor([[3.4730, 2.6554, 3.6024, 2.5813], [3.2162, 2.4591, 3.3360, 2.3904],
                         [5.0735, 3.8792, 5.2625, 3.7708]])
        print(I)

    def test_soft_L1_shrinkage(self):
        dx = torch.tensor([[-6, 3, 5, 4],[4, 1.7, 6, 7],[2.4, 8, -2, 9]]).type(self.dtype)
        thread = 3
        dx = soft_L1_shrinkage(dx, thread)
        real = torch.tensor([[-3.,  0.,  2.,  1.], [ 1.,  0.,  3.,  4.], [ 0.,  5., -0.,  6.]])
        print(dx)

    def test_psf2otf(self):
        sizeF = [4, 5]
        I = torch.tensor([[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7],[4,5,6,7,8]]).type(self.dtype)
        diff_kernelX = torch.unsqueeze(torch.tensor([1, -1]), 0).type(torch.float32)
        diff_kernelY = torch.unsqueeze(torch.tensor([1, -1]), 1).type(torch.float32)
        otfDx = psf2otf(diff_kernelX, sizeF)
        otfDy = psf2otf(diff_kernelY, sizeF)
        dxI = torch.real(torch.fft.ifft2(otfDx * torch.fft.fft2(I)))
        real_dxI = torch.tensor([[1.0000,  1.0000,  1.0000,  1.0000,  1.0000],
                                 [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000],
                                 [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000],
                                 [-3.0000, -3.0000, -3.0000, -3.0000, -3.0000]])
        real_dxI = Dux(I)
        print('real_dxI = Dux(I):', real_dxI)
        print(dxI)
        dyI = torch.real(torch.fft.ifft2(otfDy * torch.fft.fft2(I)))
        real_dyI = Duy(I)
        print('real_dyI:', real_dyI)
        real_dyI = torch.tensor([[ 1.0000,  1.0000,  1.0000,  1.0000, -4.0000],
                                 [ 1.0000,  1.0000,  1.0000,  1.0000, -4.0000],
                                 [ 1.0000,  1.0000,  1.0000,  1.0000, -4.0000],
                                 [ 1.0000,  1.0000,  1.0000,  1.0000, -4.0000]])

        print('dyI:', dyI)

    def test_getC(self):
        sizeF = [3,3]
        otfDx, otfDy, conjoDx, conjoDy, Denom1, Denom2 = getC(sizeF)
        print('otfDx:', otfDx)
        print('otfDy:', otfDy)
        print('conjoDx:', conjoDx)
        print('conjoDy:', conjoDy)
        print('Denom1:', Denom1)
        print('Denom2:', Denom2)
        err1 = torch.conj(otfDx) * otfDx - Denom1
        err2 = torch.conj(otfDy) * otfDy - Denom2
        print('err1:', err1)
        print('err2:', err2)
























