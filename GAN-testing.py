import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from main import generate_mnist_data_set, weights_init, batch_size, device

filters = 23
ker_gen = 4
image_ch = 1
enc_ker = 4

# Convention for real and fake labels during training
real_label, fake_label = 1., 0.
# Spatial size of training images, All images will be resized to this size
image_size = 64
# Size of z latent vector (i.e. size of generator input)
latent_vec_size = 10
# Size of feature maps in generator and discriminator
num_channels = 16
# number of input channels
input_channels = 1
# Number of training epochs
num_epochs = 15
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyper-param for Adam optimizers
beta1 = 0.5
# the number of steps applied to the discriminator per a single iteration of the generator
discriminator_iterations = 10


def tensor_to_plt_im(im: torch.Tensor):
    return im.permute(1, 2, 0)


class MNISTDCGen(nn.Module):
    def __init__(self):
        super(MNISTDCGen, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(latent_vec_size, filters * 8, ker_gen-2, 1, 0, bias=False),
            nn.BatchNorm2d(filters * 8),
            nn.ReLU(True),  # from 1x1 to 2x2, filter*8 channels

            nn.ConvTranspose2d(filters * 8, filters * 4, ker_gen, 2, 1, bias=False),
            nn.BatchNorm2d(filters * 4),
            nn.ReLU(True),  # from 2x2 to 4x4

            nn.ConvTranspose2d(filters * 4, filters * 2, ker_gen-1, 2, 1, bias=False),
            nn.BatchNorm2d(filters * 2),
            nn.ReLU(True),  # from 4x4 to 7x7

            nn.ConvTranspose2d(filters * 2, filters, ker_gen, 2, 1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(True),  # from 7x7 to 14x14

            nn.ConvTranspose2d(filters, image_ch, 4, 2, 1, bias=False),
            nn.Sigmoid()  # from 14x14 to 28x28
        )

    def forward(self, x):
        return self.conv(x)


class MNIST_ENC(nn.Module):
    def __init__(self):
        super(MNIST_ENC, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(image_ch, filters, enc_ker, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),  # 28x28 to 14x14

            nn.Conv2d(filters, filters * 2, enc_ker, 2, 1, bias=False),
            nn.BatchNorm2d(filters * 2),
            nn.LeakyReLU(0.2, inplace=True),  # 14x14 to 7x7

            nn.Conv2d(filters * 2, filters * 4, enc_ker-1, 2, 1, bias=False),
            nn.BatchNorm2d(filters * 4),
            nn.LeakyReLU(0.2, inplace=True),  # 7x7 to 4x4

            nn.Conv2d(filters * 4, filters * 8, enc_ker, 2, 1, bias=False),
            nn.BatchNorm2d(filters * 8),
            nn.LeakyReLU(0.2, inplace=True),  # 4x4 to 2x2

            nn.Conv2d(filters * 8, 1, enc_ker-2, 1, 0, bias=False),
            nn.Sigmoid()  # 2x2 to 1x1
        )

    def forward(self, x):
        return self.main(x).view(-1, 1)


def train(g_net: nn.Module, d_net: nn.Module):
    G_losses, D_losses, iters, gen_lst = [], [], 0, []
    optimizerD = optim.Adam(d_net.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(g_net.parameters(), lr=lr, betas=(beta1, 0.999))
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            d_net.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            real_label = torch.ones(b_size)
            fake_label = torch.zeros(b_size)
            # Forward pass real batch through D
            optimizerD.zero_grad()
            real_output = d_net(real_cpu).view(-1)
            errD_real = criterion(real_output, real_label)
            # errD_real.backward()
            D_x = real_output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, latent_vec_size, 1, 1, device=device)
            # Generate fake image batch with G
            fake = g_net(noise)
            # Classify all fake batch with D
            fake_output = d_net(fake.detach()).view(-1)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD = criterion(fake_output, fake_label)
            errD.backward()
            errD_real.backward()
            # errD_fake.backward()
            D_G_z1 = fake_output.mean().item()
            # Update D
            optimizerD.step()

            # perform a generator iteration every 'discriminator_iterations' steps of the discriminator
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            g_net.zero_grad()
            optimizerG.zero_grad()
            # Since we just updated D, perform another forward pass of all-fake batch through D
            fake_output = d_net(fake).view(-1)
            # Calculate G's loss based on this output
            # errG = criterion(fake_output, label)
            errG = criterion(fake_output, fake_label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = fake_output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item()+errD_real.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            G_losses.append(errG.item())
            D_losses.append(errD.item()+errD_real.item())  # Save Losses for plotting later

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = g_net(fixed_noise).detach().cpu()
                gen_lst.append(vutils.make_grid(fake))
                plt.imshow(tensor_to_plt_im(gen_lst[-1]))
                plt.show()
            iters += 1
    torch.save(d_net.state_dict(), './d_net')
    torch.save(g_net.state_dict(), './g_net')


def test_generator(path):
    """
    this function tests a pre-trained generator by feeding it with random vectors form the latent space
    and showing the output images
    """
    g_net = MNISTDCGen()
    g_net.load_state_dict(torch.load(path))
    g_net.eval()
    gen_lst = []
    for i in range(10000):
        manualSeed = random.randint(1, 10000)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        fixed_noise = torch.randn(image_size, latent_vec_size, device=device)
        with torch.no_grad():
            fake = g_net(fixed_noise).detach().cpu()
        gen_lst.append(vutils.make_grid(fake))
        plt.imshow(tensor_to_plt_im(gen_lst[-1]))
        plt.show()
        time.sleep(5)


if __name__ == '__main__':

    dataloader = generate_mnist_data_set()
    # generator = MNISTGen()
    # discriminator = MNISTDisc(output_dim=1)
    gen = MNISTDCGen()
    d = MNIST_ENC()
    criterion = nn.BCELoss()

    # generator.apply(weights_init), discriminator.apply(weights_init)
    gen.apply(weights_init), d.apply(weights_init)

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(image_size, latent_vec_size, 1, 1, device=device)
    # fixed_noise = torch.randn(image_size, latent_vec_size, 1, 1, device=device)
    # train(generator, discriminator, d_loss, gen_loss_non_saturating)
    train(gen, d)
