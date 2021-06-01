import random
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def generate_mnist_data_set():
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)


device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

# Batch size during training
batch_size = 32
# Size of z latent vector (i.e. size of generator input)
latent_vec_size = 10
# Size of feature maps in generator and discriminator
num_channels = 32
# number of input channels
input_channels = 1
# Number of training epochs
num_epochs = 5
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyper-param for Adam optimizers
beta1 = 0.5
# the number of steps applied to the discriminator per a single iteration of the generator
discriminator_iterations = 1


def tensor_to_plt_im(im: torch.Tensor):
    return im.permute(1, 2, 0)


def d_loss(discriminator_generated_x, discriminator_true_x):
    """
    cross-entropy loss
    """
    return -0.5 * torch.mean(torch.log(discriminator_true_x + 1e-8)) \
           - 0.5 * torch.mean(torch.log(1 - discriminator_generated_x + 1e-8))


def gen_loss(discriminator_generated_x, discriminator_true_x):
    return -1 * d_loss(discriminator_generated_x, discriminator_true_x)


def gen_loss_non_saturating(discriminator_generated_x):
    """
    non-saturating cross-entropy loss
    """
    return -0.5 * torch.mean(torch.log(discriminator_generated_x + 1e-8))


def d_loss_least_squares(discriminator_generated_x, discriminator_true_x):
    """
    least squares loss (discriminator)
    """
    return 0.5 * torch.mean(torch.square(discriminator_true_x + 1e-8 - 1)) \
           + 0.5 * torch.mean(torch.square(discriminator_generated_x + 1e-8))


def gen_loss_least_squares(discriminator_generated_x):
    """
    least squares loss (generator)
    """
    return 0.5 * torch.mean(torch.square(discriminator_generated_x + 1e-8 - 1))


class MNISTGen(nn.Module):
    def __init__(self):
        super(MNISTGen, self).__init__()
        self.conv = nn.Sequential(
            # input is (latent_vec_size) x 1 x 1
            nn.ConvTranspose2d(latent_vec_size, num_channels * 8, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channels * 8),
            nn.ReLU(True),
            # input is (num_channels*8) x 2 x 2
            nn.ConvTranspose2d(num_channels * 8, num_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_channels * 4),
            nn.ReLU(True),
            # size is (num_channels*4) x 4 x 4
            nn.ConvTranspose2d(num_channels * 4, num_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_channels * 2),
            nn.ReLU(True),
            # size (num_channels*2) x 7 x 7
            nn.ConvTranspose2d(num_channels * 2, num_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(True),
            # size (num_channels) x 14 x 14
            nn.ConvTranspose2d(num_channels, input_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.Tanh()  # the final output image size. (num_input_channels) x 28 x 28
        )
        self.linear = nn.Sequential(
            nn.Linear(latent_vec_size, (num_channels * 8) * 2 * 2),
        )

    def forward(self, x):
        return self.conv(x)


class MNISTDisc(nn.Module):
    def __init__(self, output_dim):
        super(MNISTDisc, self).__init__()
        self.conv = nn.Sequential(
            # input size is (num_input_channels) x 28 x 28
            nn.Conv2d(input_channels, num_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # size (num_channels) x 14 x 14
            nn.Conv2d(num_channels, num_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # size (num_channels*2) x 7 x 7
            nn.Conv2d(num_channels * 2, num_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # size (num_channels*4) x 4 x 4
            nn.Conv2d(num_channels * 4, num_channels * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # size (num_channels*4) x 2 x 2
            nn.Conv2d(num_channels * 8, 1, kernel_size=2, bias=False),
            nn.Sigmoid()  # size (1 x 1 x 1)
        )
        self.linear = nn.Sequential(
            nn.Linear((num_channels * 8) * 2 * 2, output_dim),
        )

    def forward(self, x):
        return self.conv(x).view(-1, 1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(g_net: nn.Module, d_net: nn.Module, criterion=nn.BCELoss()):
    G_losses, D_losses, iters, gen_lst = [], [], 0, []
    optimizerD = optim.Adam(d_net.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(g_net.parameters(), lr=lr, betas=(beta1, 0.999))
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            # train discriminator
            d_net.zero_grad()
            batch = data[0].to(device)
            b_size = batch.size(0)
            # Forward pass real batch through D
            real_output = d_net(batch).view(-1)
            real_label = torch.ones((b_size,), dtype=torch.float, device=device)
            errD_real = criterion(real_output, real_label)
            errD_real.backward()

            # Train with fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, latent_vec_size, 1, 1, device=device)
            # Generate fake image batch with generator
            fake = g_net(noise)
            # Classify all fake batch with discriminator
            fake_output = d_net(fake.detach()).view(-1)
            fake_label = torch.ones((b_size,), dtype=torch.float, device=device)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake = criterion(fake_output, fake_label)
            errD_fake.backward()
            # Update discriminator
            optimizerD.step()

            # perform a generator iteration every 'discriminator_iterations' steps of the discriminator
            # Update generator
            g_net.zero_grad()
            # perform forward pass of fake batch through discriminator
            fake_output = d_net(fake).view(-1)
            real_label = torch.ones((b_size,), dtype=torch.float, device=device)
            # Calculate G's loss based on this output
            errG = criterion(fake_output, real_label)
            errG.backward()
            # Update generator weights
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                      % (epoch+1, num_epochs, i, len(dataloader), (errD_real.item() + errD_fake.item())/2, errG.item()))
            G_losses.append(errG.item())
            D_losses.append((errD_real.item() + errD_fake.item()) / 2)  # Save Losses for plotting later

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or i == len(dataloader) - 1:
                with torch.no_grad():
                    fixed_noise = torch.randn(2 * batch_size, latent_vec_size, 1, 1, device=device)
                    fake = g_net(fixed_noise).detach().cpu()
                gen_lst.append(vutils.make_grid(fake))
                plt.imshow(tensor_to_plt_im(gen_lst[-1]))
                plt.show()
            iters += 1
    torch.save(d_net.state_dict(), './d_net1')
    torch.save(g_net.state_dict(), './g_net1')


def test_generator(path, num_tests):
    """
    this function tests a pre-trained generator by feeding it with random vectors form the latent space
    and showing the output images
    """
    g_net = MNISTGen()
    g_net.load_state_dict(torch.load(path))
    g_net.eval()
    gen_lst = []
    for i in range(num_tests):
        manualSeed = random.randint(1, 10000)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        noise = torch.randn(2 * batch_size, latent_vec_size, 1, 1, device=device)
        with torch.no_grad():
            fake = g_net(noise).detach().cpu()
        gen_lst.append(fake)
        plt.imshow(tensor_to_plt_im(gen_lst[-1][-1]), cmap='gray')
        plt.show()
        time.sleep(2)


if __name__ == '__main__':
    # test_generator('./g_net', 20)
    dataloader = generate_mnist_data_set()

    generator = MNISTGen().to(device)
    discriminator = MNISTDisc(output_dim=1).to(device)

    generator.apply(weights_init), discriminator.apply(weights_init)

    # Create batch of latent vectors to check the generator's progress
    train(generator, discriminator)
