'''
Cho-TMLR from the following paper:
@inproceedings{chen2025riemgcp,
  title={Understanding Matrix Function Normalizations in Covariance Pooling through the Lens of Riemannian Geometry},
  author={Ziheng Chen and Yue Song and Xiaojun Wu and Gaowen Liu and Nicu Sebe},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
}
'''
import torch
import torch.nn as nn
from torch.autograd import Function
# from .feature_momentum import *

class LCMCOV(nn.Module):
     """Cho-TMLR

     Args:
         iterNum: #iteration of Newton-schulz method
         is_sqrt: whether perform matrix square root or not
         is_vec: whether the output is a vector or not
         input_dim: the #channel of input feature
         dimension_reduction: if None, it will not use 1x1 conv to
                               reduce the #channel of feature.
                              if 256 or others, the #channel of feature
                               will be reduced to 256 or others.
     """
     def __init__(self, is_vec=True, input_dim=2048, dimension_reduction=None,eps=1e-6):

         super(LCMCOV, self).__init__()
         self.is_vec = is_vec
         self.dr = dimension_reduction
         # self.feature_scaling = feature_scaling()
         if self.dr is not None:
             self.conv_dr_block = nn.Sequential(
               nn.Conv2d(input_dim, self.dr, kernel_size=1, stride=1, bias=False),
               nn.BatchNorm2d(self.dr),
               nn.ReLU(inplace=True)
             )
         output_dim = self.dr if self.dr else input_dim
         if self.is_vec:
             self.output_dim = int(output_dim*(output_dim+1)/2)
         else:
             self.output_dim = int(output_dim*output_dim)
         self._init_weight()
         tmp = torch.eye(dimension_reduction, dtype=torch.float64) if dimension_reduction != None else torch.eye(input_dim, dtype=torch.float64)
         self.register_buffer('I', tmp)
         self.eps=eps
     def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

     def _cov_pool(self, x):
         return Covpool.apply(x)

     def _triuvec(self, x):
         return Triuvec.apply(x)

     def LogI_LCM(self,x):
         x_permuted = x + self.eps * self.I
         L = torch.linalg.cholesky(x_permuted)
         strict_tril = L.tril(-1)
         diag_part = torch.diag_embed(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)))
         tril_matrix = strict_tril + diag_part
         return tril_matrix + tril_matrix.transpose(-1, -2)

     def forward(self, x):
         if self.dr is not None:
             x = self.conv_dr_block(x)
         x = self._cov_pool(x)
         x = self.LogI_LCM(x)
         if self.is_vec:
             x = self._triuvec(x)
         return x

     def __repr__(self):
         return f"{self.__class__.__name__}(is_vec={self.is_vec}," \
                f" input_dim={self.input_dim}, dimension_reduction={self.dr},eps={self.eps})"



class Covpool(Function):
     @staticmethod
     def forward(ctx, input):
         input = input.double()
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         I_hat = (-1./M/M)*torch.ones(M,M,device = x.device) + (1./M)*torch.eye(M,M,device = x.device)
         I_hat = I_hat.view(1,M,M).repeat(batchSize,1,1).type(x.dtype)
         y = x.bmm(I_hat).bmm(x.transpose(1,2))
         ctx.save_for_backward(input,I_hat)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,I_hat = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         grad_input = grad_output + grad_output.transpose(1,2)
         grad_input = grad_input.bmm(x).bmm(I_hat)
         #print(grad_input.mean())
         grad_input = grad_input.reshape(batchSize,dim,h,w).float()
         return grad_input

class Triuvec(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         x = x.reshape(batchSize, dim*dim)
         I = torch.ones(dim,dim).triu().reshape(dim*dim)
         index = I.nonzero()
         y = torch.zeros(batchSize,int(dim*(dim+1)/2),device = x.device).type(dtype)
         y = x[:,index].float()
         ctx.save_for_backward(input,index)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,index = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         grad_input = torch.zeros(batchSize,dim*dim,device = x.device,requires_grad=False)
         grad_input[:,index] = grad_output
         grad_input = grad_input.reshape(batchSize,dim,dim).type(dtype)
         return grad_input

def CovpoolLayer(var):
    return Covpool.apply(var)

