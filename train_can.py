import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from torch.utils.tensorboard import SummaryWriter

from parameters import *
from dataloader import *
from model_CAN import *

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.backends.cudnn.benchmark = True

# Create the dataloader
dataloader = get_dataset()

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Create the generator
netG = Generator().to(device)
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Initialize BCELoss function
criterion = nn.BCELoss()
criterion_style = nn.CrossEntropyLoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup SGD optimizers for both G and D
#optimizerD = optim.SGD(netD.parameters(), lr=lr, momentum=0.9)
#optimizerG = optim.SGD(netG.parameters(), lr=lr, momentum=0.9)
# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

################################################################################
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
entropies = []
n_iter = 0
writer = SummaryWriter('./logs/')

# Plot some training images
real_batch = next(iter(dataloader))
real_img_sample = vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu()
writer.add_image('Sample Real Images', real_img_sample)

################################################################################
# Training Loop
print("Starting Training Loop...")

# For each epoch
for epoch in range(num_epochs):
    netG.train()
    netD.train()
    torch.set_grad_enabled(True)
    # For each batch in the dataloader
    for i, (data, style_label) in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        #target = target.to(device)
        style_label = style_label.to(device)
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output, output_style = netD(real_cpu)
        output = output.view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        errD_real = errD_real + criterion_style(output_style.squeeze(), style_label.squeeze())
        # Calculate gradients for D in backward pass
        #errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output, output_style = netD(fake.detach())
        output = output.view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        #errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        errD.backward()
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output, output_style = netD(fake)
        output = output.view(-1)
        # Uniform cross entropy
        logsoftmax = nn.LogSoftmax(dim=1)
        unif = torch.full((data.shape[0], n_class), 1/n_class)
        unif = unif.to(device)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        errG = errG + torch.mean(-torch.sum(unif * logsoftmax(output_style), 1))
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        style_entropy = -1 * (nn.functional.softmax(output_style, dim=1) * nn.functional.log_softmax(output_style, dim=1))
        style_entropy = style_entropy.sum(dim=1).mean() / torch.log(torch.tensor(n_class).float())

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.3f\tLoss_G: %.3f\tD(x): %.3f\tD(G(z)): %.3f / %.3f\t Entropy: %.3f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, style_entropy))

        writer.add_scalar('Generator loss', errG.item(), n_iter)
        writer.add_scalar('Discriminator loss', errD.item(), n_iter)
        n_iter += 1

    # Save model and generated images
    if epoch % 5 == 0:
        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()

        writer.add_image('Generated Image (Epoch: ' + str(epoch) + ')', vutils.make_grid(fake, padding=2, normalize=True))

        torch.save(netG.state_dict(), 'logs/model_can_'+str(epoch)+'.pth')

    torch.save(netG.state_dict(), 'logs/model_can.pth')

writer.close()
