import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from AutoEncoderACELEB import AutoEncoderACELEB, generate_celeba64_data_set
from GAN import MNISTGen, generate_mnist_data_set, tensor_to_plt_im

AE_PATH = './auto_encoder'
GEN_PATH = './g_net'  # path of the pre-trained generator
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

# Number of training iterations
num_iter = 5000
# Learning rate for optimizers
lr = 0.02
# Beta1 hyper-param for Adam optimizers
beta1 = 0.5
# the batch index that will determine each image from the dataloader batch will be chosen (default is 0)
index = 10
# interpolation coefficient
a = 0.5
# dimension of the latent space
latent_vec_size = 10
image_size = 28  # equals 28 or 64


def model_inversion(image):
    """
    this function performs a model inversion with a pre-determined GAN. i.e. given an image, finds the corresponding
    vector in the latent space by minimizing the loss function defined as: 'original_image-generated_image', where
    generated_image is the output image created by the generator with the current latent-space vector as input (Q 1.2)
    """
    z = torch.autograd.Variable(torch.randn(1, latent_vec_size, 1, 1).
                                type(torch.FloatTensor), requires_grad=True).to(device)
    criterion = nn.MSELoss()
    for i in range(num_iter):
        generated_image = g_net(z).view(image.shape)
        loss = criterion(generated_image, image)
        loss.backward()
        z.retain_grad()
        z.data -= lr * z.grad.data
        if i % 1000 == 0:
            print(f"loss: {loss}")
    return z


def generate_masks():
    points = np.arange(0, image_size, 1)
    xs, ys = np.meshgrid(points, points)
    mask1 = np.where(xs % 2 == 0, 1, 0) + np.where(ys % 2 == 0, 1, 0)
    mask2 = np.where(xs % 3 == 1, 0, 1) + np.where(ys % 3 == 1, 0, 1)
    mask3 = np.where(xs % 4 == 1, 0, 1) + np.where(ys % 4 == 1, 0, 1)
    return mask1, mask2, mask3


def restore_image(im: torch.Tensor):
    """
    this function performs image restoration on the given image by applying three different masks
    on it and performing model inversion with a pre-trained GAN model over the masked image. The
    return value is the three different vectors from the latent space that are generated by the
    model inversion process described above (Q 1.3)
    """
    mask1, mask2, mask3 = generate_masks()
    mask1, mask2, mask3 = torch.from_numpy(mask1).type(torch.int).to(device), \
                          torch.from_numpy(mask2).type(torch.int).to(device), \
                          torch.from_numpy(mask3).type(torch.int).to(device)
    return model_inversion(torch.mul(im, mask1)), model_inversion(torch.mul(im, mask2)), \
           model_inversion(torch.mul(im, mask3))


def image_interpolation(z1, z2, gen: nn.Module, im_size, latent_size):
    """
    this function performs image interpolation between the images that are generated by setting the given
    latent-vectors as input to the given generator (which is created by using GAN or Auto-Encoder) and shows
    the result
    """
    z1, z2 = z1.view(1, latent_size, 1, 1), z2.view(1, latent_size, 1, 1)
    im1, im2 = gen(z1).view(im_size, im_size, 1).detach(), gen(z2).view(im_size, im_size, 1).detach()
    print("image number one:\n")
    plt.imshow(im1)
    plt.show()
    print("image number two:\n")
    plt.imshow(im2)
    plt.show()
    print(f"interpolated image with a={a}:\n")
    plt.imshow(im1 * a + im2 * (1 - 0.5))
    plt.show()


if __name__ == '__main__':
    # model inversion for images form the mnist data-set
    g_net = MNISTGen()
    g_net.load_state_dict(torch.load(GEN_PATH))
    g_net.eval()
