import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

from torchvision import datasets, transforms
from mnist_model import Generator, Discriminator, DHead, QHead
from utils import *

batch_size = 100

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

netG = Generator().to(device)
netG.apply(weights_init)
print(netG)

discriminator = Discriminator().to(device)
discriminator.apply(weights_init)
print(discriminator)

netD = DHead().to(device)
netD.apply(weights_init)
print(netD)

netQ = QHead().to(device)
netQ.apply(weights_init)
print(netQ)

criterionD = nn.BCELoss()
criterionQ_dis = nn.CrossEntropyLoss()
criterionQ_con = NormalNLLLoss()

optimD = optim.Adam([{'params': discriminator.parameters()}, {'params': netD.parameters()}], lr=0.0002, betas=(0.5, 0.999))
optimG = optim.Adam([{'params': netG.parameters()}, {'params': netQ.parameters()}], lr=0.0002, betas=(0.5, 0.999))

# Fixed Noise
z = torch.randn(100, 62, 1, 1, device=device)

idx = np.arange(10).repeat(10)
dis_c = torch.zeros(100, 10, 1, 1, device=device)
dis_c[torch,arange(0, 100), idx] = 1

con_c = torch.Tensor(100, 2, 1, 1, device=device).uniform(-1, 1)

fixed_noise = torch.cat((z, dis_c, con_c), dim=1)


real_label = 1
fake_label = 0

print("-"*25)
print("Starting Training Loop...\n")
#print('Epochs: %d\nBatch Size: %d\nLength of Data Loader: %d' % (params['epoch_num'], params['batch_size'], len(train_loader)))
print("-"*25)

start_time = time.time()