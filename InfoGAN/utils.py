import torch.nn as nn

def weights_init(m):
	if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif(type(m) == nn.BatchNorm2d):
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)