
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import os

if not os.path.exists('outputs'):
    os.makedirs('outputs')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # run on GPU if possible
batch_size = 100

# defining the dataset transformation (normalization)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
to_pil_image = transforms.ToPILImage()

# loading the dataset
train_data = datasets.MNIST(
    root='../input/data',
    train=True,
    download=True,
    transform=transform
)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


def save_generator_image(image, path):
    save_image(image, path)


# ENC Hyper-Parameters
latent_size = 100
cnv2Filters = 16
q_pixel = 7
kernel = 4
pixels = 28
end_pixels = int((((pixels - 3) / 2) - 3) / 2)  # two iters of 1-strid conv+pool2
ker_gen = 2
depth = 10


class AE(nn.Module):
    def __init__(self, latent_size):
        super(AE, self).__init__()
        self.latent_size = latent_size
        self.conv1 = nn.Conv2d(1, cnv2Filters, kernel)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(cnv2Filters, cnv2Filters * 2, kernel)
        self.fc1E = nn.Linear(cnv2Filters * 2 * end_pixels * end_pixels,
                              self.latent_size)
        self.fc1D = nn.Linear(self.latent_size, q_pixel ** 2 * depth)  # FC layer with enough nodes to represent
        # a low resolution image (quarter of 28*28)
        # * depth times
        self.Tconv1 = nn.ConvTranspose2d(depth, int(cnv2Filters / 2), ker_gen, stride=(2, 2))
        self.Tconv2 = nn.ConvTranspose2d(int(cnv2Filters / 2), cnv2Filters, ker_gen,
                                         stride=(2, 2))
        self.conv = nn.Conv2d(cnv2Filters, 1, 1)
        self.fc1E_bn = nn.BatchNorm1d(self.latent_size)
        self.fc1_bn = nn.BatchNorm1d(q_pixel ** 2 * depth)
        self.Tconv1_bn = nn.BatchNorm2d(int(cnv2Filters / 2))
        self.Tconv2_bn = nn.BatchNorm2d(cnv2Filters)

    def encoder(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, cnv2Filters * 2 * end_pixels * end_pixels)
        x = F.relu(self.fc1E_bn(self.fc1E(x)))
        return x

    def decoder(self, x):
        x = F.relu(self.fc1_bn(self.fc1D(x)))  # latent_Size to 7*7
        x = x.view(-1, depth, q_pixel, q_pixel)  # depthx7x7 tensor
        x = F.relu(self.Tconv1_bn(self.Tconv1(x)))  # 7 to 14 and #filters from depth to cnv2filters/2
        x = torch.sigmoid(self.Tconv2_bn(self.Tconv2(x)))  # 14 to 28 and #filters to cnv2filters
        x = torch.sigmoid(self.conv(x))  # 28*28 single channel
        return x

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x


AE = AE(latent_size).to(device)


def get_moments(d):
    # Return moments of the data provided
    mean = torch.mean(d)
    diffs = d - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    kurtoses = torch.mean(torch.pow(zscores, 4.0))
    return mean, var, kurtoses

criterion = nn.MSELoss()

def net_loss(net_output, encoder_output, img):
    """
    loss function to minimize both the final MSE and latent space
    distance from normal distribution 
    """
    im_loss = criterion(net_output, img)
    mean, var, kurtoses = get_moments(encoder_output)
    stat_loss = (mean - 0)**2+(var - 1)**2+(kurtoses - 3)**2
    loss = im_loss + stat_loss
    return loss


images = []
# optimizers - Adam is a variation of the SGD
optimizer = optim.Adam(AE.parameters(), lr=0.002)
epochs = 10


# function to train the network
def train(train_loader, epochs):
    train_loss = []
    for epoch in range(epochs):
        running_loss = 0.0
        for data in train_loader:
            img, _ = data
            img = img.to(device)

            optimizer.zero_grad()
            net_output = AE(img)
            encoder_output = AE.encoder(img)
            loss = net_loss(net_output, encoder_output, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        generated_img = make_grid(net_output)
        # save the generated torch tensor models to disk
        save_generator_image(img, f"outputs/orig_img{epoch}.png")
        save_generator_image(generated_img, f"outputs/gen_img{epoch}.png")
        images.append(generated_img)
        loss = running_loss / len(train_loader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch + 1, epochs, loss))
    return train_loss


train_loss = train(train_loader, epochs)


def latentVector(batch_size,latent_size):
    """
    function for creating a random latent vector
    :param latent_size: vector size
    :param batch_size: how many different vectors in 1 batch
    :return:
    """
    return torch.randn(batch_size, latent_size).to(device)

z = latentVector(batch_size,latent_size)
outputs = AE.decoder(z)

novel_img = make_grid(outputs)
# save the generated torch tensor models to disk
save_generator_image(novel_img, "outputs/novel_img.png")




