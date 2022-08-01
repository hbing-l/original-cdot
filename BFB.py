import matplotlib.pyplot as plt
import ot
from ot.bregman import sinkhorn
import torch

import numpy as np

torch.set_default_tensor_type('torch.DoubleTensor')


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda:1')
    else:
        return torch.device('cpu')


device = get_default_device()


def prox_sinkhorn(P, C, r, c, lr, tau, it):
    # P contains the gradient step
    """Uses logsumexp for numerical stability."""

    with torch.no_grad():
        A = (- C * lr + P) / (1 + lr * tau)
        logr = torch.log(r).reshape(-1, 1)
        logc = torch.log(c).reshape(1, -1)

        for i in range(it):
            A = A + logr - A.logsumexp(dim=1, keepdim=True)
            A = A + logc - A.logsumexp(dim=0, keepdim=True)

        res = torch.exp(A)

    return res


def optimize_BFB(x, y, C, tau, it, epochs, lr, seed=42, verbose=True):
    # NumPy -> PyTorch
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    C = torch.from_numpy(C).to(device)

    # Initialization
    torch.manual_seed(seed)
    P = C.clone()  # torch.rand(n, n, requires_grad=True)

    # Optimization
    history = []
    for epoch in range(epochs + 1):

        # -- Gradient Step --
        grad_step = torch.log(P.data)  # + lr * P.grad.data
        # grad_step[grad_step<=0] = 1e-5

        # Proximity operator
        P.data = prox_sinkhorn(grad_step, C, x, y, lr, tau, it)

        # Tracking
        total_cost = torch.sum(P.data * C + tau * P.data * torch.log(P.data) - tau * P.data)  # + cost
        history.append(total_cost.item())
        if verbose and (epoch == 0 or epoch % 100 == 0):
            print('[Epoch %4d/%d] loss: %f' % (epoch, epochs, total_cost.item()))

    # PyTorch -> NumPy
    P = P.squeeze()
    P = P.detach().cpu().numpy()

    # Convergence plot
    plt.plot(history)
    plt.show()
    return P


def optimize_BFB_timereg(x, y, C, tau, timreg, reg_norm, gammak_old, Xtk, Xt_old, it, epochs, lr, seed=42,
                         verbose=True):
    # NumPy -> PyTorch
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    C = torch.from_numpy(C).to(device)
    gammak_old = torch.from_numpy(gammak_old).to(device)
    Xtk = torch.from_numpy(Xtk).to(device)
    Xt_old = torch.from_numpy(Xt_old).to(device)

    # Initialization
    torch.manual_seed(seed)
    P = C.clone().requires_grad_()  # torch.rand(n, n, requires_grad=True)

    # Optimization
    history = []
    for epoch in range(epochs + 1):

        # -- Gradient Step --
        if reg_norm == 'mixed':
            J = timreg * torch.sum(torch.sqrt(torch.sum((P @ Xtk - gammak_old @ Xt_old) ** 2, dim=1)))
        else:
            J = timreg * torch.sum(((P @ Xtk - gammak_old @ Xt_old) ** 2))

        J.backward()
        grad = P.grad.data
        grad_step = torch.log(P.data) - lr * grad

        # Proximity operator
        P.data = prox_sinkhorn(grad_step, C, x, y, lr, tau, it)
        P.grad.data.zero_()

        # Tracking
        total_cost = torch.sum(P.data * C + tau * P.data * torch.log(P.data) - tau * P.data) + J
        history.append(total_cost.item())
        if verbose and (epoch == 0 or epoch % 100 == 0):
            print('[Epoch %4d/%d] loss: %f' % (epoch, epochs, total_cost.item()))

    # PyTorch -> NumPy
    P = P.squeeze()
    P = P.detach().cpu().numpy()
    plt.plot(history)
    plt.show()
    return P


def gcg_proximal(a, b, M, lr, reg1, dh, f, df, G0=None, numItermax=10,
        numInnerItermax=200, stopThr=1e-9, verbose=False, log=False):
    
    loop = 1
    
    if log:
        log = {'loss': []}

    if G0 is None:
        G = np.outer(a, b)
    else:
        G = G0
        
    def cost(G):
        return np.sum(M * G) + reg1 * np.sum(G * np.log(G) - G) + f(G)

    f_val = cost(G)
    if log:
        log['loss'].append(f_val)

    it = 0

    if verbose:
        print('{:5s}|{:12s}|{:8s}'.format(
            'It.', 'Loss', 'Delta loss') + '\n' + '-' * 32)
        print('{:5d}|{:8e}|{:8e}'.format(it, f_val, 0))

    while loop:

        it += 1
        #print(it)
        old_fval = f_val

        # problem linearization
        Mi = lr*M - dh(G) + lr * df(G)

        # solve linear program with Sinkhorn
        # Gc = sinkhorn_stabilized(a,b, Mi, reg1, numItermax = numInnerItermax)
        G = sinkhorn(a, b, Mi, 1 + lr * reg1, numItermax=numInnerItermax)

        # test convergence
        if it >= numItermax:
            loop = 0
            
        f_val = cost(G)
        
        delta_fval = (f_val - old_fval) / abs(f_val)
        if abs(delta_fval) < stopThr:
            loop = 0

        if log:
            log['loss'].append(f_val)

        if verbose:
            if it % 20 == 0:
                print('{:5s}|{:12s}|{:8s}'.format(
                    'It.', 'Loss', 'Delta loss') + '\n' + '-' * 32)
            print('{:5d}|{:8e}|{:8e}'.format(it, f_val, delta_fval))

    if log:
        return G, log
    else:
        return G
