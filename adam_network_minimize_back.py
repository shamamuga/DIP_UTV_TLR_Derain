from models import fcn
import torch
import torch.optim
from utils.commom_utils import *


def adam_network_minmize(net, net_affine, net_input_saved, net_affine_input_saved, y, R, V, Lam, mu, new_size, num_iter):
    LR = 0.01
    reg_noise_std = 0.01
    dtype = torch.float32
    # Loss
    # Losses
    mse1 = torch.nn.MSELoss().type(dtype)
    mse2 = torch.nn.MSELoss().type(dtype)

    optimizer = torch.optim.Adam([{'params': net.parameters()}, {'params': net_affine.parameters(), 'lr': 1e-4}], lr=LR)
    for step in range(num_iter):
        print('adam_network_minmize: step num:', step)
        # net_input = net_input_saved + reg_noise_std * torch.zeros(net_input_saved.shape).type_as(net_input_saved.data).normal_()
        net_input = net_input_saved
        net_affine_input = net_affine_input_saved
        # net_input_affine = net_input_affine_saved + reg_noise_std*torch.zeros(net_input_affine_saved.shape).type_as(net_input_affine_saved.data).normal_()
        optimizer.zero_grad()
        out_x = net(net_input).squeeze()
        # out_x = out_x.squeeze()
        out_a = net_affine(net_affine_input)
        out_a_m = out_a.view(2, 2)
        out_a_m_p = F.pad(input=out_a_m, pad=(0, 1, 0, 0), mode='constant', value=0)
        # print(out_a_m)
        # out_y = imtransform(y, out_a_m)
        out_y = trans_crop(y, out_a_m_p, new_size)
        print('adam_network_minmize:out_y.size()', out_y.size())
        dx_out_x = Dux(out_x)
        # total_loss = mse(out_y - out_x, R) + mu*mse(dx_out_x, V - Lam/mu)
        loss1 = mse1(out_y - out_x, R)
        loss2 = mu*mse2(dx_out_x, V - Lam/mu)
        loss1.backward(retain_graph=True)
        loss2.backward()
        # total_loss = loss1 + loss2
        # total_loss.backward()
        optimizer.step()
    print('adam_network_minmize: out_x.size()', out_x.size())
    return out_x, out_a_m





