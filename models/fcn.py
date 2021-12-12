import torch
import torch.nn as nn


def fcn(num_input_channels=200, num_output_channels=1, num_hidden=1000):

    model = nn.Sequential()
    model.add_module('Layer1', nn.Linear(num_input_channels, num_hidden, bias=True))
    model.add_module('Inner_Relu6', nn.ReLU6())

    model.add_module('Layer2', nn.Linear(num_hidden, num_output_channels, bias=True))
    model.add_module('Outer_Relu', nn.ReLU())
    #   model.add(nn.Softmax())
    return model
