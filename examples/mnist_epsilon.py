import waitGPU
waitGPU.wait(utilization=20, available_memory=10000, interval=10)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from convex_adversarial import DualNetBounds, robust_loss

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import setproctitle
import argparse

import problems as pblm

class Metric:
    def __init__(self, epsilon):
        self.mean = 0
        self.min = 1e10
        self.max = 0
        self.sum = 0
        self.n = 0
        self.verified = 0
        self.eps = epsilon
    
    def add(self, e):
        e = max(e, 0)
        self.n += 1
        self.sum += e
        self.min = min(self.min, e)
        self.max = max(self.max, e)
        self.mean = self.sum / self.n
        self.verified += e >= self.eps + 1e-4
    
    def __str__(self):
        return "==========\nMean: %.4f,\tMin: %.4f,\tMax: %.4f,\tVerified: %d/%d==========\n" % (self.mean, self.min, self.max, self.verified, self.n)
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--niters', type=int, default=20)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--threshold', type=float, default=1e-3)
    parser.add_argument('--prefix', default='paper')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--bound_type', type=str, default="paper")

    parser.add_argument('--mnist', action='store_true')
    parser.add_argument('--svhn', action='store_true')
    parser.add_argument('--har', action='store_true')
    parser.add_argument('--fashion', action='store_true')

    args = parser.parse_args()
    prefix = 'mnist_conv_{:.4f}_{:.4f}_0'.format(args.epsilon, 1e-3).replace(".","_") + args.prefix
    filename = prefix + "_model.pth"

    if args.mnist: 
        train_loader, test_loader = pblm.mnist_loaders(args.batch_size)
        model = pblm.mnist_model().cuda()
        model.load_state_dict(torch.load(filename))
    elif args.svhn: 
        train_loader, test_loader = pblm.svhn_loaders(args.batch_size)
        model = pblm.svhn_model().cuda()
        model.load_state_dict(torch.load('svhn_new/svhn_epsilon_0_01_schedule_0_001'))
    elif args.har:
        pass
    elif args.fashion: 
        pass
    else:
        raise ValueError("Need to specify which problem.")
    for p in model.parameters(): 
        p.requires_grad = False

    num_classes = model[-1].out_features

    correct = []
    incorrect = []
    l = []

    loader = train_loader if args.train else test_loader
    m = Metric(args.epsilon)
    for j,(X,y) in enumerate(loader): 
        print('*** Batch {} ***'.format(j))
        epsilon = Variable(args.epsilon*torch.ones(args.batch_size).cuda(), requires_grad=True)
        X, y = Variable(X).cuda(), Variable(y).cuda()

        out = Variable(model(X).data.max(1)[1])
        idx = y==out
        if (idx.sum().item() == 0):
            continue
        X = X[idx]
        y = y[idx]
        out = out[idx]

        # form c without the 0 row
        c = Variable(torch.eye(num_classes).type_as(X.data)[out.data].unsqueeze(1) - torch.eye(num_classes).type_as(X.data).unsqueeze(0))
        I = (~(out.data.unsqueeze(1) == torch.arange(num_classes).type_as(out.data).unsqueeze(0)).unsqueeze(2))
        c = (c[I.repeat(1, 1, num_classes)].view(X.size(0),num_classes-1,num_classes))
        if X.is_cuda:
            c = c.cuda()

        alpha = args.alpha

        def f(eps): 
            dual = DualNetBounds(model, X, eps.unsqueeze(1), True, True, args.bound_type)
            f = -dual.g(c)
            return (f.max(1)[0])
           
        l_eps = torch.zeros(c.size(0)).cuda()
        u_eps = torch.ones(c.size(0)).cuda()
        for i in range(args.niters): 
            mid = (l_eps + u_eps) / 2
            f_max = f(mid)
            if (f_max.data.abs() <= args.threshold).all(): 
                break
            l_eps = torch.where(f_max <= 0, mid, l_eps)
            u_eps = torch.where(f_max <= 0, u_eps, mid)

            del f_max
#         print(mid, f(mid))
        if (f(mid).data.abs() > args.threshold).any(): 
            print(j)
#         correct.append(epsilon[y==out])
        for e in mid:
            m.add(e.item())
#         incorrect.append(epsilon[y!=out])

        del X, y
    print(str(m))
    print(l)
#     torch.save(torch.cat(correct, 0), '{}_correct_epsilons.pth'.format(prefix+args.bound_type))
#     torch.save(torch.cat(incorrect, 0), '{}_incorrect_epsilons.pth'.format(prefix+args.bound_type))