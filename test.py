import numpy
import scipy
import matplotlib
import torch
a = torch.Tensor([1.])
from torch.backends import cudnn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.is_acceptable(a.to(device))