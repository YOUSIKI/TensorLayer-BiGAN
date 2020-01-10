import os
import math
import numpy as np
import tensorlayer as tl
import tensorflow as tf
from tensorlayer.layers import (Input, Dense, DeConv2d, Reshape, BatchNorm2d, Conv2d, Flatten, Dropout, Concat)
from data import flags, get_celebA
from model import get_generator, get_encoder

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def test(weights_pathG, weights_pathE, real_path='real_image.png', reproduced_path='reproduced_image.png'):
    images, images_path = get_celebA(flags.output_size, flags.n_epoch, flags.batch_size)
    num_tiles = int(math.ceil(math.sqrt(flags.sample_size)))
    G = get_generator([None, flags.z_dim])
    G.load_weights(weights_pathG, format='npz')
    G.eval()
    E = get_encoder([None, flags.output_size, flags.output_size, flags.c_dim])
    E.load_weights(weights_pathE, format='npz')
    E.eval()

    for step, batch_images in enumerate(images):
        if batch_images.shape[0] != flags.batch_size:
            break
        result = G(E(batch_images))
        tl.visualize.save_images(batch_images.numpy(), [num_tiles, num_tiles], real_path)
        tl.visualize.save_images(result.numpy(), [num_tiles, num_tiles], reproduced_path)
        break


if __name__ == "__main__":
    test('checkpoint/G.npz', 'checkpoint/E.npz')