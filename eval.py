import os
import math
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from model import get_generator


def eval(weights_path, image_path='samples.png', samples=64, x_shape=[None, 64, 64, 3], z_shape=[None, 128]):
    num_tiles = int(math.ceil(math.sqrt(samples)))
    G = get_generator(x_shape, z_shape)
    G.load_weights(weights_path, format='npz')
    G.eval()
    z_shape[0] = samples
    z = np.random.normal(loc=0.0, scale=1.0, size=z_shape).astype(np.float32)
    tl.visualize.save_images(G(z).numpy(), [num_tiles, num_tiles], image_path)


if __name__ == '__main__':
    eval('checkpoints/10-G.npz')
