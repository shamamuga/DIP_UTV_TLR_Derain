import unittest
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from models.fcn import *
from utils.commom_utils import *
from models.unet import UNet
from utils.network_optimization import *


class TestCommon_utils(unittest.TestCase):
    def setUp(self):
        pass

    def test_affine_network_optimization(self):
        dtype = torch.float32
        net_input_affine = get_noise(4, 'noise', [1, 1]).type(dtype)
        net_input_affine = torch.tensor([[[0.0496, 0.0768, 0.0088, 0.0132]]])
        net_input_affine.squeeze_()
        print('net_input_affine', net_input_affine)
        net_affine_input_saved = net_input_affine.detach().clone()
        net_affine = fcn(num_input_channels=4, num_output_channels=1, num_hidden=6).type(dtype)
        out_a = net_affine(net_affine_input_saved)
        O = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1],
                          [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]])
        O = np_to_torch(O)
        O = O[None, ...].type(dtype)
        target_affine = torch.tensor([[0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1],
                                      [0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1 ]]).type(dtype)
        target_affine = O
        num_iter = 500
        new_size = O.size()
        theta_initial = net_affine(net_affine_input_saved)
        print('theta_initial(before affine_network_optimization):', theta_initial)

        out_O = affine_network_optimization(net_affine, net_affine_input_saved, O, target_affine, num_iter, new_size)
        print('out_O directly return from affine_network_optimization:', out_O)
        theta = net_affine(net_affine_input_saved)
        print('theta = net_affine(net_affine_input_saved):', theta)

    def test_cnn_network_optimization(self):
        dtype = torch.float32
        INPUT = 'noise'
        INPUT = 'meshgrid'
        input_depth = 1
        net_input = get_noise(input_depth+1, INPUT, (64, 64)).type(dtype)
        print(net_input.size())
        net_input = net_input[:, 0, :, :]
        net_input = net_input.unsqueeze(1)
        print(net_input.size())
        print(net_input)
        net_input_saved = net_input.detach().clone()
        net = UNet(num_input_channels=input_depth, num_output_channels=1, upsample_model='deconv', need_sigmoid=False)
        num_iter = 1000
        target_net = torch.zeros(64, 64)
        net_input = net_input.squeeze()
        target_net1 = Dux(net_input)
        # target_net1 = target_net[None, :]
        target_net1 = target_net1.detach()
        target_net2 = net_input
        # target_net2 = target_net2[None, :]
        target_net2 = target_net2.detach()
        out_x_before = net(net_input_saved)
        out_x = cnn_network_optimization(net, net_input_saved, target_net1, target_net2, num_iter, 0.3)
        print('before train out_x_before:', out_x_before)
        print('out_x return out_x = cnn_network_optimization', out_x)
        out_x_after = net(net_input_saved)
        print('out_x after train:', out_x_after)
        print('origin net_input', net_input)
        print('target net1:', target_net1)
        print('target_net2:', target_net2)
        err = out_x_after - target_net2
        print('err:', err)
        # print('the values of the parmeters of the model:')
        ''' for name, param in net.named_parameters():
            if param.requires_grad:
                print(name, param.data) '''









