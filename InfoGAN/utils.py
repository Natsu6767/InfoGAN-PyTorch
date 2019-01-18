import torch
import torch.nn as nn
import numpy as np

def weights_init(m):
	if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif(type(m) == nn.BatchNorm2d):
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)

def NormalNLLLoss(x, mu, var):
	"""
	Calculate the negative log likelihood
	of normal distribution.
	This needs to be minimised.

	Treating Q(cj | x) as a factored Gaussian.
	"""
	logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
	nll = -(logli.sum(1).mean())

	return nll

def noise_sampler(dis_c, con_c, noise, batch_size):

	idx = np.random.randint(10, size=batch_size)
	c = np.zeros((batch_size, 10))
	c[range(batch_size), idx] = 1.0

	dis_c.data.copy_(torch.Tensor(c))
	con_c.data.uniform_(-1.0, 1.0)
	noise.data.normal_(0.0, 1.0)

