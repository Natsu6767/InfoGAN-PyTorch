import argparse

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', required=True, help='Checkpoint to load path from')
args = parser.parse_args()

from models.celeba_model import Generator

def mix(i, m, n):

    c = torch.zeros(50, 10, 10, device=device)
    for j in range(10):
        c[torch.arange(0, 50), j, i[0]] = 1.0

    c[torch.arange(0, 50), m-1] = 0.0
    c[torch.arange(0, 50), m-1, i[1]] = 1.0

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


idx = np.zeros((2, 50))
idx[0] = np.array([1, 2, 5, 7, 9]).repeat(10)
idx[1] = np.tile(np.arange(10), 5)



z = torch.randn(50, 128, 1, 1, device=device)


k = 0
for a in range(10):
    for b in range(10):
        k += 1

        c = mix(idx, a+1, b+1)
        noise = torch.cat((z, c), dim=1)

        with torch.no_grad():
            generated_img1 = netG(noise).detach().cpu()
        # Display the generated image.
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(generated_img1, nrow=10, padding=2, normalize=True), (1,2,0)))
        plt.savefig('celebatests/{}'.format(k))
        print('Saved_{}'.format(k))
