import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from data import get_celebA
from model import (
    get_encoder,
    get_generator,
    get_discriminator
)
from config import flags


def train():
    num_tiles = int(math.ceil(math.sqrt(flags.sample_size)))
    images, images_path = get_celebA(flags.output_size, flags.batch_size)
    n_step_epoch = int(len(images_path) // flags.batch_size)
    x_shape = [None, flags.output_size, flags.output_size, flags.c_dim]
    z_shape = [None, flags.z_dim]

    E = get_encoder(x_shape, z_shape)
    G = get_generator(x_shape, z_shape)
    D = get_discriminator(x_shape, z_shape)

    if flags.load_weights:
        E.load_weights(flags.load_weights % 'E', format='npz')
        G.load_weights(flags.load_weights % 'G', format='npz')
        D.load_weights(flags.load_weights % 'D', format='npz')
    
    E.train()
    G.train()
    D.train()
    
    d_optimizer = tf.optimizers.Adam(flags.lr, beta_1=flags.beta1)
    g_optimizer = tf.optimizers.Adam(flags.lr, beta_1=flags.beta1)
    
    for epoch in range(flags.n_epoch):
        for step, batch_images in enumerate(images):
            if batch_images.shape[0] != flags.batch_size: # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                # z = Z.sample([flags.batch_size, flags.z_dim]) 
                z_fake = np.random.normal(loc=0.0, scale=1.0, size=[flags.batch_size, flags.z_dim]).astype(np.float32)
                x_fake = G(z_fake)
                x_real = batch_images
                z_real = E(x_real)
                y_fake = D([x_fake, z_fake])
                y_real = D([x_real, z_real])
                d_loss_real = tl.cost.sigmoid_cross_entropy(y_real, tf.ones_like(y_real), name='d_loss_real')
                d_loss_fake = tl.cost.sigmoid_cross_entropy(y_fake, tf.zeros_like(y_fake), name='d_loss_fake')
                d_loss = d_loss_real + d_loss_fake
                g_loss_real = tl.cost.sigmoid_cross_entropy(y_real, tf.zeros_like(y_real), name='g_loss_real')
                g_loss_fake = tl.cost.sigmoid_cross_entropy(y_fake, tf.ones_like(y_fake), name='g_loss_fake')
                g_loss = g_loss_real + g_loss_fake
            grad = tape.gradient(g_loss, G.trainable_weights + E.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights + E.trainable_weights))
            grad = tape.gradient(d_loss, D.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
            del tape

            print("Epoch: [{}/{}] [{}/{}] took: {:.3f}, d_loss: {:.5f}, g_loss: {:.5f}".format(epoch, \
                  flags.n_epoch, step, n_step_epoch, time.time()-step_time, d_loss, g_loss))
        
        if np.mod(epoch, flags.save_every_epoch) == 0:
            E.save_weights(f'{flags.checkpoint_dir}/{epoch}-E.npz', format='npz')
            G.save_weights(f'{flags.checkpoint_dir}/{epoch}-G.npz', format='npz')
            D.save_weights(f'{flags.checkpoint_dir}/{epoch}-D.npz', format='npz')
            G.eval()
            tl.visualize.save_images(G(z_fake).numpy(), [num_tiles, num_tiles], '{}/train_{:02d}.png'.format(flags.sample_dir, epoch))
            G.train()


if __name__ == '__main__':
    train()
