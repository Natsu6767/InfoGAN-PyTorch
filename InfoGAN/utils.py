import torch.nn as nn

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
	"""
	logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
	nll = -(logli.sum(1).mean())

	return nll