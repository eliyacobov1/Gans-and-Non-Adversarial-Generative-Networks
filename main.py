import numpy as np
from skimage.color import rgb2gray
from imageio import imread
from matplotlib import pyplot as plt


def read_image(filename, representation):
    im = imread(filename)

    return rgb2gray(im) if representation == 1 \
        else im.astype(np.float64) / 255

im1 = read_image('./output/num6.png', 0)
im2 = read_image('./output/num9.png', 0)

for i in range(2, 13):
    plt.imshow(im1*(1/i) + im2*(1-(1/i)), cmap='gray')
    plt.show()
z = 0