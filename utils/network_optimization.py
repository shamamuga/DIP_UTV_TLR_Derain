import torch.optim
from utils.commom_utils import *


def affine_network_optimization(net_affine, net_affine_input_saved, O, target, num_iter, new_size):
    LR = 0.01
    LR = 0.001
    dtype = torch.float32
    # net_affine.register_backward_hook(hook_fn_backward)
    # Loss
    mse = torch.nn.MSELoss().type(dtype)
    optimizer = torch.optim.Adam([{'params': net_affine.parameters()}], lr=LR)
    for step in range(num_iter):
        # print('affine_network_optimization: step', step)
        optimizer.zero_grad()
        out_a = net_affine(net_affine_input_saved)
        # print('in step {step} in affine_network_optimization out_a.requires_grad'.format(step=step), out_a)
        # out_a_m = out_a.view(2, 2)

        # out_a_m = torch.tensor([[torch.cos(out_a), -torch.sin(out_a)], [torch.sin(out_a), torch.cos(out_a)]], requires_grad=True)
        out_a_m = torch.stack((torch.cat((torch.cos(out_a), -torch.sin(out_a))), torch.cat((torch.sin(out_a), torch.cos(out_a)))))
        # print('in affine_network_optimization out_a_m:', out_a_m)
        out_a_m_p = F.pad(input=out_a_m, pad=(0, 1, 0, 0), mode='constant', value=0)
        # out_a_m_p.retain_grad()
        out_O = trans_crop(O, out_a_m_p, new_size)
        # out_O.retain_grad()
        loss = mse(out_O, target)
        # print('in step {step} in affine_network_optimization loss'.format(step=step), loss)
        # loss.retain_grad()
        loss.backward()
        # print('in step {step} in affine_network_optimization out_a_m_p.grad'.format(step=step), out_a_m_p.grad)
        # print('in step {step} in affine_network_optimization out_O.grad'.format(step=step), out_O.grad)
        # print('in step {step} in affine_network_optimization loss.grad'.format(step=step), loss.grad)
        optimizer.step()
    out_a = net_affine(net_affine_input_saved)
    # print('affine_network_optimization:', out_a)
    # out_a_m = out_a.view(2, 2)
    out_a_m = torch.tensor([[torch.cos(out_a), -torch.sin(out_a)], [torch.sin(out_a), torch.cos(out_a)]])
    out_a_m_p = F.pad(input=out_a_m, pad=(0, 1, 0, 0), mode='constant', value=0)
    out_O = trans_crop(O, out_a_m_p, new_size)
    return out_O


def cnn_network_optimization(net, net_input_saved, target1, target2, num_iter, mu):
    LR = 0.006
    LR = 0.002
    dtype = torch.float32
    '''modules = net.named_children()
    for name, module in modules:
        module.register_backward_hook(hook_fn_backward)'''
    # net.register_backward_hook(hook_fn_backward)
    # Loss
    mse = torch.nn.MSELoss().type(dtype)
    #mse2 = torch.nn.MSELoss().type(dtype)
    optimizer = torch.optim.Adam([{'params': net.parameters()}], lr=LR)
    for step in range(num_iter):
        optimizer.zero_grad()
        out_b = net(net_input_saved).squeeze()
        #target2 = target2[None, :]
        # out_b = out_b.squeeze()
        dx_out_b = Dux(out_b)
        loss = mse(out_b, target2) + mu * mse(dx_out_b, target1)

        print('in step {step} in cnn_network_optimization loss'.format(step=step), loss)
        # loss = mse(out_b, target)
        loss.backward()
        optimizer.step()
    out_b = net(net_input_saved)
    return out_b


def hook_fn_backward(module, grad_input, grad_output):
    # print(module)
    # 为了符合反向传播的顺序，我们先打印 grad_output
    print('grad_output', grad_output)
    # 再打印 grad_input
    print('grad_input', grad_input)




