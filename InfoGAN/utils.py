import torch
import torch.nn as nn
import numpy as np

def weights_init(m):
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif(type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.

    Treating Q(cj | x) as a factored Gaussian.
    """
    def __call__(x, mu, var):
        
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll

def noise_sampler(n_dis_c, dis_c_dim, n_con_c, n_z, batch_size, device):

    z = torch.randn(batch_size, n_z, 1, 1, device=device)

    dis_c = torch.zeros(batch_size, n_dis_c, dis_c_dim, device=device)
    idx = torch.zeros(n_dis_c, batch_size, device=device)
    
    for i in range(n_dis_c):
        idx[i] = torch.randint(low=0, high=dis_c_dim, size=(batch_size))
        c[torch.arange(0, batch_size), i, idx[i]] = 1.0

    dis_c.squeeze_()
    if batch_size == 1:
        dis_c.unsqueeze_(0)

    dis_c = dis_c.view(batch_size, -1, 1, 1)

    con_c = torch.Tensor(batch_size, n_con_c, 1, 1, device=device).uniform(-1, 1)

    noise = torch.cat((z, dis_c, con_c), dim=1)

    return noise, idx