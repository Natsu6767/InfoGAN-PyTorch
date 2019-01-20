import argparse

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', required=True, help='Checkpoint to load path from')
args = parser.parse_args()

from models.svhn_model import Generator

def mix1(i, m):

    c = torch.zeros(50, 4, 10, device=device)
    for j in range(4):
        c[torch.arange(0, 50), j, i[0]] = 1.0

    c[torch.arange(0, 50), m-1] = 0.0
    c[torch.arange(0, 50), m-1, i[1]] = 1.0

    return c.view(50, -1, 1, 1)

def mix2(i, m, n):

    c = torch.zeros(50, 4, 10, device=device)
    for j in range(4):
        c[torch.arange(0, 50), j, 0] = 1.0

    c[torch.arange(0, 50), m-1] = 0.0
    c[torch.arange(0, 50), m-1, i[0]] = 1.0

    c[torch.arange(0, 50), n-1] = 0.0
    c[torch.arange(0, 50), n-1, i[1]] = 1.0

    return c.view(50, -1, 1, 1)


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

c = np.linspace(-1, 1, 10).reshape(1, -1)
c = np.repeat(c, 5, 0).reshape(-1, 1)
c = torch.from_numpy(c).float().to(device)
c = c.view(50, 1, 1, 1)

zeros = torch.zeros(50, 1, 1, 1, device=device)

c5 = torch.cat((c, zeros, zeros, zeros), dim=1)
c6 = torch.cat((zeros, c, zeros, zeros), dim=1)
c7 = torch.cat((zeros, zeros, c, zeros), dim=1)
c8 = torch.cat((zeros, zeros, zeros, c), dim=1)
c10 = torch.cat((c, c, c, c), dim=1)

idx = np.zeros((2, 50))
idx[0] = np.arange(3, 8).repeat(10)
idx[1] = np.tile(np.arange(0, 10), 5)

"""
dis_c = torch.zeros(50, 4, 10, device=device)

for i in range(4):
    dis_c[torch.arange(0, 50), i, idx[0]] = 1.0

#dis_c[torch.arange(0, 50), i, idx[0]] = 1.0

dis_c = dis_c.view(50, -1, 1, 1)
"""
#c = mix2(idx, 4, 1)
dis_c = mix1(idx, 4)

z = torch.randn(50, 124, 1, 1, device=device)

#noise = torch.cat((z, c, c5), dim=1)

noise = torch.cat((z, dis_c, c6), dim=1)
"""
noise2 = torch.cat((z, dis_c, c6), dim=1)
noise3 = torch.cat((z, dis_c, c7), dim=1)
noise4 = torch.cat((z, dis_c, c8), dim=1)
"""
with torch.no_grad():
    generated_img1 = netG(noise).detach().cpu()
# Display the generated image.
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(generated_img1, nrow=10, padding=2, normalize=True), (1,2,0)))
plt.show()
"""
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
"""