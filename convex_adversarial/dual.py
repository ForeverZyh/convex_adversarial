import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
# import cvxpy as cp

from . import affine as Aff

def AffineTranspose(l): 
    if isinstance(l, nn.Linear): 
        return Aff.AffineTransposeLinear(l)
    elif isinstance(l, nn.Conv2d): 
        return Aff.AffineTransposeConv2d(l)
    else:
        raise ValueError("AffineTranspose class not found for given layer.")

def Affine(l): 
    if isinstance(l, nn.Linear): 
        return Aff.AffineLinear(l)
    elif isinstance(l, nn.Conv2d): 
        return Aff.AffineConv2d(l)
    else:
        raise ValueError("Affine class not found for given layer.")

def full_bias(l, n=None): 
    # expands the bias to the proper size. For convolutional layers, a full
    # output dimension of n must be specified. 
    if isinstance(l, nn.Linear): 
        return l.bias.view(1,-1)
    elif isinstance(l, nn.Conv2d): 
        b = l.bias.unsqueeze(1).unsqueeze(2)
        k = int((n/(b.numel()))**0.5)
        return b.expand(b.numel(),k,k).contiguous().view(1,-1)
    else:
        raise ValueError("Full bias can't be formed for given layer.")

def batch(A, n): 
    return A.view(n, -1, *A.size()[1:])
def unbatch(A): 
    return A.view(-1, *A.size()[2:])

def forward_linear(_coef_lb, _coef_ub, _bias_lb, _bias_ub, W, b):
    coef_lb = torch.matmul(torch.clamp(W, min=0), _coef_lb) + torch.matmul(torch.clamp(W, max=0), _coef_ub)
    coef_ub = torch.matmul(torch.clamp(W, min=0), _coef_ub) + torch.matmul(torch.clamp(W, max=0), _coef_lb)
    bias_lb = torch.matmul(_bias_lb, torch.clamp(W, min=0).t()) + torch.matmul(_bias_ub, torch.clamp(W, max=0).t()) + b
    bias_ub = torch.matmul(_bias_ub, torch.clamp(W, min=0).t()) + torch.matmul(_bias_lb, torch.clamp(W, max=0).t()) + b

    return coef_lb, coef_ub, bias_lb, bias_ub

def forward_activation(_coef_lb, _coef_ub, _bias_lb, _bias_ub, lb, ub):
    # lb, ub [batch, last_dim]
    # _coef_lb, _coef_ub [batch, last_dim, input]
    # _bias_lb, _bias_ub [batch, last_dim]
    lb_coef_pre = torch.where(lb >= 0, torch.ones_like(lb),
                              torch.where(ub <= 0,
                                          torch.zeros_like(lb),
                                          torch.where(-lb < ub,
                                                      torch.ones_like(lb),
                                                      torch.zeros_like(lb))))
    ub_coef_pre = torch.where(lb >= 0, torch.ones_like(lb),
                              torch.where(ub <= 0,
                                          torch.zeros_like(lb),
                                          ub / (ub - lb)))
    
    # lb_coef_pre, ub_coef_pre [batch, last_dim]
    lb_cst_pre = torch.zeros_like(lb)
    ub_cst_pre = torch.where((lb >= 0) | (ub <= 0), torch.zeros_like(lb),
                             -lb * ub / (ub - lb))
    # lb_cst_pre, ub_cst_pre [batch, last_dim]
    
    coef_lb = _coef_lb * torch.clamp(lb_coef_pre, min=0).unsqueeze(-1) + _coef_ub * torch.clamp(lb_coef_pre, max=0).unsqueeze(-1)
    coef_ub = _coef_ub * torch.clamp(ub_coef_pre, min=0).unsqueeze(-1) + _coef_lb * torch.clamp(ub_coef_pre, max=0).unsqueeze(-1)
    bias_lb = _bias_lb * torch.clamp(lb_coef_pre, min=0) + _bias_ub * torch.clamp(lb_coef_pre, max=0) + lb_cst_pre
    bias_ub = _bias_ub * torch.clamp(ub_coef_pre, min=0) + _bias_lb * torch.clamp(ub_coef_pre, max=0) + ub_cst_pre
    return coef_lb, coef_ub, bias_lb, bias_ub

class DualNetBounds: 
    def __init__(self, net, X, epsilon, alpha_grad=False, scatter_grad=False, bound_type="paper"):
        n = X.size(0)

        self.layers = [l for l in net if isinstance(l, (nn.Linear, nn.Conv2d))]
        self.affine_transpose = [AffineTranspose(l) for l in self.layers]
        self.affine = [Affine(l) for l in self.layers]
        self.k = len(self.layers)+1

        # initialize affine layers with a forward pass
        _ = X[0].view(1,-1)
        for a in self.affine: 
            _ = a(_)
            
        self.biases = [full_bias(l, self.affine[i].out_features) 
                        for i,l in enumerate(self.layers)]
        
        gamma = [self.biases[0]]
        nu = []
        nu_hat_x = self.affine[0](X)

        eye = Variable(torch.eye(self.affine[0].in_features)).type_as(X)
        nu_hat_1 = self.affine[0](eye).unsqueeze(0)

        l1 = (nu_hat_1).abs().sum(1)
        
        self.zl = [nu_hat_x + gamma[0] - epsilon*l1]
        self.zu = [nu_hat_x + gamma[0] + epsilon*l1]
        if bound_type == "LBP":
            coef_lb = torch.eye(self.affine[0].in_features).unsqueeze(0).repeat(n, 1, 1).cuda()
            coef_ub = coef_lb.clone()
            bias_lb = torch.zeros((n, self.affine[0].in_features)).cuda()
            bias_ub = bias_lb.clone()
            coef_lb, coef_ub, bias_lb, bias_ub = forward_linear(coef_lb, coef_ub, bias_lb, bias_ub, self.affine[0].l.weight, self.affine[0].l.bias)
            

        self.I = []
        self.I_empty = []
        self.I_neg = []
        self.I_pos = []
        self.X = X
        self.epsilon = epsilon
        I_collapse = []
        I_ind = []

        # set flags
        self.scatter_grad = scatter_grad
        self.alpha_grad = alpha_grad

        for i in range(0,self.k-2):
            if bound_type == "LBP": # ReLU
                # compute coef_lb, coef_ub, bias_lb, bias_ub for i+1-th layer
                coef_lb, coef_ub, bias_lb, bias_ub = forward_activation(coef_lb, coef_ub, bias_lb, bias_ub, self.zl[-1], self.zu[-1])
            # compute sets and activation
            self.I_neg.append((self.zu[-1] <= 0).detach())
            self.I_pos.append((self.zl[-1] > 0).detach())
            self.I.append(((self.zu[-1] > 0) * (self.zl[-1] < 0)).detach())
            self.I_empty.append(self.I[-1].data.long().sum() == 0)
            
            I_nonzero = ((self.zu[-1]!=self.zl[-1])*self.I[-1]).detach()
            d = self.I_pos[-1].type_as(X).clone()

            # Avoid division by zero by indexing with I_nonzero
            if I_nonzero.data.sum() > 0:
                d[I_nonzero] += self.zu[-1][I_nonzero]/(self.zu[-1][I_nonzero] - self.zl[-1][I_nonzero])

            # indices of [example idx, origin crossing feature idx]
            I_ind.append(Variable(self.I[-1].data.nonzero()))
            
            # initialize new terms
            if not self.I_empty[-1]:
                out_features = self.affine[i+1].out_features

                subset_eye = Variable(X.data.new(self.I[-1].data.sum(), d.size(1)).zero_())
                subset_eye.scatter_(1, I_ind[-1][:,1,None], d[self.I[-1]][:,None])

                if not scatter_grad: 
                    subset_eye = subset_eye.detach()
                nu.append(self.affine[i+1](subset_eye))

                # create a matrix that collapses the minibatch of origin-crossing indices 
                # back to the sum of each minibatch
                I_collapse.append(Variable(X.data.new(I_ind[-1].size(0), X.size(0)).zero_()))
                I_collapse[-1].scatter_(1, I_ind[-1][:,0][:,None], 1)
            else:
                nu.append(None)         
                I_collapse.append(None)
            gamma.append(self.biases[i+1])
            # propagate terms
            gamma[0] = self.affine[i+1](d * gamma[0])
            for j in range(1,i+1):
                gamma[j] = self.affine[i+1](d * gamma[j])
                if not self.I_empty[j-1]: 
                    nu[j-1] = self.affine[i+1](d[I_ind[j-1][:,0]] * nu[j-1])

            nu_hat_x = self.affine[i+1](d*nu_hat_x)
            nu_hat_1 = batch(self.affine[i+1](unbatch(d.unsqueeze(1)*nu_hat_1)), n)
            
            l1 = (nu_hat_1).abs().sum(1)
            
            # compute bounds
            if bound_type == "paper":
                self.zl.append(nu_hat_x + sum(gamma) - epsilon*l1 + 
                               sum([(self.zl[j][self.I[j]] * (-nu[j].t()).clamp(min=0)).mm(I_collapse[j]).t()
                                    for j in range(i+1) if not self.I_empty[j]]))
                self.zu.append(nu_hat_x + sum(gamma) + epsilon*l1 - 
                               sum([(self.zl[j][self.I[j]] * nu[j].t().clamp(min=0)).mm(I_collapse[j]).t()
                                    for j in range(i+1) if not self.I_empty[j]]))
            elif bound_type == "IBP":
                l = self.affine[i+1](F.relu(self.zl[-1]))
                u = self.affine[i+1](F.relu(self.zu[-1]))
                self.zl.append(torch.min(l, u) + self.biases[i+1])
                self.zu.append(torch.max(l, u) + self.biases[i+1])
            elif bound_type == "LBP":
                coef_lb, coef_ub, bias_lb, bias_ub = forward_linear(coef_lb, coef_ub, bias_lb, bias_ub, self.affine[i+1].l.weight, self.affine[i+1].l.bias)
                input_lb = (self.X.view(n, -1) - epsilon).view(n, 1, -1)
                input_ub = (self.X.view(n, -1) + epsilon).view(n, 1, -1)
                lb = torch.sum(torch.clamp(coef_lb, min=0) * input_lb + torch.clamp(coef_lb, max=0) * input_ub, dim=-1) + bias_lb 
                ub = torch.sum(torch.clamp(coef_ub, min=0) * input_ub + torch.clamp(coef_ub, max=0) * input_lb, dim=-1) + bias_ub 
                self.zl.append(lb)
                self.zu.append(ub)
                
        
        self.s = [torch.zeros_like(u) for l,u in zip(self.zl, self.zu)]

        for (s,l,u) in zip(self.s,self.zl, self.zu): 
            I_nonzero = (u != l).detach()
            if I_nonzero.data.sum() > 0: 
                s[I_nonzero] = u[I_nonzero]/(u[I_nonzero]-l[I_nonzero])

        
    def g(self, c):
        n = c.size(0)
        nu = [[]]*self.k
        nu[-1] = -c
        for i in range(self.k-2,-1,-1):
            nu[i] = batch(self.affine_transpose[i](unbatch(nu[i+1])),n)
            if i > 0:
                # avoid in place operation
                out = nu[i].clone()
                i_neg_index = self.I_neg[i-1].unsqueeze(1).repeat(1, out.size(1), 1)
                i_index = self.I[i-1].unsqueeze(1).repeat(1, out.size(1), 1)
                out[i_neg_index] = 0
                if not self.I_empty[i-1]:
                    if self.alpha_grad: 
                        out[i_index] = (self.s[i-1].unsqueeze(1).expand(*nu[i].size())[i_index] * 
                                                               nu[i][i_index])
                    else:
                        out[i_index] = ((self.s[i-1].unsqueeze(1).expand(*nu[i].size())[i_index] * 
                                                                                   torch.clamp(nu[i], min=0)[i_index])
                                                         + (self.s[i-1].detach().unsqueeze(1).expand(*nu[i].size())[i_index] * 
                                                                                   torch.clamp(nu[i], max=0)[i_index]))
                nu[i] = out

        f = (-sum(nu[i+1].matmul(self.biases[i].view(-1)) for i in range(self.k-1))
             -nu[0].matmul(self.X.view(self.X.size(0),-1).unsqueeze(2)).squeeze(2)
             -self.epsilon*nu[0].abs().sum(2)
             + sum((nu[i].clamp(min=0)*self.zl[i-1].unsqueeze(1)).matmul(self.I[i-1].type_as(self.X).unsqueeze(2)).squeeze(2) 
                    for i in range(1, self.k-1) if not self.I_empty[i-1]))
           
        return f

def robust_loss(net, epsilon, X, y, 
                size_average=True, alpha_grad=False, scatter_grad=False, bound_type="paper"):
    num_classes = net[-1].out_features
    dual = DualNetBounds(net, X, epsilon, alpha_grad, scatter_grad, bound_type)
    c = Variable(torch.eye(num_classes).type_as(X.data)[y.data].unsqueeze(1) - torch.eye(num_classes).type_as(X.data).unsqueeze(0))
    if X.is_cuda:
        c = c.cuda()
    f = -dual.g(c)
    err = (f.data.max(1)[1] != y.data)
    if size_average: 
        err = err.sum().item()/X.size(0)
    ce_loss = nn.CrossEntropyLoss(size_average=size_average)(f, y)
    return ce_loss, err