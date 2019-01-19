import argparse

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', required=True, help='Checkpoint to load path from')
args = parser.parse_args()

import sys
sys.path.append('/home/mohit/DeepLearning/InfoGAN/')
from models.celeba_model import Generator

def mix(c, i, m, n):

    c[torch.arange(0, 100), m-1, 0] = 0.0
    c[torch.arange(0, 100), m-1, i[0]] = 1.0

    c[torch.arange(0, 100), n-1, 0] = 0.0
    c[torch.arange(0, 100), n-1, i[1]] = 1.0

    return c.view(100, -1, 1, 1)

# Load the checkpoint file
state_dict = torch.load(args.load_path)

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

# Create the generator network.
netG = Generator().to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['netG'])
print(netG)

idx = np.zeros((2, 100))
idx[0] = np.arange(10).repeat(10)
idx[1] = np.tile(np.arange(10), 10)

dis_c = torch.zeros(100, 10, 10, device=device)
for i in range(10):
    dis_c[torch.arange(0, 100), i, 0] = 1.0

c1c2 = mix(dis_c, idx, 1, 2)
c2c3 = mix(dis_c, idx, 2, 3)
c5c6 = mix(dis_c, idx, 5, 6)
c9c10 = mix(dis_c, idx, 9, 10)

z = torch.randn(100, 128, 1, 1, device=device)

noise1 = torch.cat((z, c1c2), dim=1)
noise2 = torch.cat((z, c2c3), dim=1)
noise3 = torch.cat((z, c5c6), dim=1)
noise4 = torch.cat((z, c9c10), dim=1)

with torch.no_grad():
    generated_img1 = netG(noise1).detach().cpu()
# Display the generated image.
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(generated_img1, nrow=10, padding=2, normalize=True), (1,2,0)))
plt.show()

with torch.no_grad():
    generated_img2 = netG(noise2).detach().cpu()
# Display the generated image.
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(generated_img2, nrow=10, padding=2, normalize=True), (1,2,0)))
plt.show()

with torch.no_grad():
    generated_img3 = netG(noise3).detach().cpu()
# Display the generated image.
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(generated_img3, nrow=10, padding=2, normalize=True), (1,2,0)))
plt.show()

with torch.no_grad():
    generated_img4 = netG(noise4).detach().cpu()
# Display the generated image.
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(generated_img4, nrow=10, padding=2, normalize=True), (1,2,0)))
plt.show()