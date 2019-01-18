import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random

from mnist_model import Generator, Discriminator, DHead, QHead
from dataloader import get_data
from utils import *

# Set random seed for reproducibility.
seed = 1123
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

batch_size = 100
epochs = 10

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

dataloader = get_data('MNIST', batch_size)

# Plot the training images.
sample_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(
    sample_batch[0].to(device)[ : 100], padding=2, normalize=True).cpu(), (1, 2, 0)))

plt.show()

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

img_list = []
G_losses = []
D_losses = []

print("-"*25)
print("Starting Training Loop...\n")
#print('Epochs: %d\nBatch Size: %d\nLength of Data Loader: %d' % (params['epoch_num'], params['batch_size'], len(train_loader)))
print("-"*25)

start_time = time.time()
iters = 0

for epoch in range(epochs):
    epoch_start_time = time.time()

    for i, (data, _) in enumerate(dataloader, 0):
        # Get batch size
        b_size = data.size(0)
        # Transfer data tensor to GPU/CPU (device)
        real_data = data.to(device)

        optimD.zero_grad()
        label = torch.full((b_size, ), real_label, device=device)
        output1 = discriminator(real_data)
        probs_real = netD(output1)
        loss_real = criterionD(probs_real, label)
        loss_real.backward()

        # Fake Data
        label.fill_(fake_label)
        noise, idx = noise_sample(1, 10, 2, 62, batch_size, device)
        fake_data = netG(noise)
        output2 = discriminator(fake_data)
        probs_fake = netD(output2)
        loss_fake = criterionD(probs_fake, label)
        loss_fake.backward()

        # Net Loss for the discriminator
        D_loss = loss_real + loss_fake
        # Update parameters
        optimD.step()

        # Updating G and Q
        optimG.zero_grad()

        output = discriminator(fake_data)
        label.fill_(real_label)
        probs_fake = netD(output)
        gen_loss = criterionD(probs_fake, label)

        q_logits, q_mu, q_var = netQ(output)
        target = idx
        dis_loss = 0
        for i in range(1):
            dis_loss += criterionQ_dis(q_logits[:, i*10 + i*10 + 10], target[i])

        con_loss = criterionQ_con(con_c, q_mu, q_var)*0.1

        G_loss = gen_loss + dis_loss + con_loss
        G_loss.backward()

        optimG.step()

        # Check progress of training.
        if i != 0 and i%100 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch+1, epochs, i, len(dataloader), 
                    D_loss.item(), G_loss.item()))

        # Save the losses for plotting.
        G_losses.append(G_loss.item())
        D_losses.append(D_loss.item())

        # Check how the generator is doing by saving G's output on a fixed noise.
        if (iters % 100 == 0) or ((epoch == epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                gen_data = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(gen_data, padding=2, normalize=True))

        iters += 1

    epoch_time = time.time() = epoch_start_time
    print("Time taken for Epoch %d: %.2fs" %(epoch + 1, epoch_time))

training_time = time.time() - start_time
print("-"*50)
print('Training finished!\nTotal Time for Training: %.2fm' %(training_time / 60))
print("-"*50)


# Plot the training losses.
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Animation showing the improvements of the generator.
fig = plt.figure(figsize=(10,10))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
plt.show()
anim.save('infoGAN.gif', dpi=80, writer='imagemagick')