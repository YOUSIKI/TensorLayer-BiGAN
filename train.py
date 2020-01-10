import os, time, multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from glob import glob
from data import get_celebA, flags
from model import get_generator, get_discriminator, get_encoder

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
num_tiles = int(np.sqrt(flags.sample_size))

def train():
    images, images_path = get_celebA(flags.output_size, flags.n_epoch, flags.batch_size)
    G = get_generator([None, flags.z_dim])
    D = get_discriminator([None, flags.z_dim], [None, flags.output_size, flags.output_size, flags.c_dim])
    E = get_encoder([None, flags.output_size, flags.output_size, flags.c_dim])

    if flags.load_weights:
        E.load_weights('checkpoint/E.npz', format='npz')
        G.load_weights('checkpoint/G.npz', format='npz')
        D.load_weights('checkpoint/D.npz', format='npz')

    G.train()
    D.train()
    E.train()

    d_optimizer = tf.optimizers.Adam(flags.lr, beta_1=flags.beta1)
    g_optimizer = tf.optimizers.Adam(flags.lr, beta_1=flags.beta1)
    e_optimizer = tf.optimizers.Adam(flags.lr, beta_1=flags.beta1)

    n_step_epoch = int(len(images_path) // flags.batch_size)

    for epoch in range(flags.n_epoch):
        for step, batch_images in enumerate(images):
            if batch_images.shape[0] != flags.batch_size:
                break
            step_time = time.time()

            with tf.GradientTape(persistent=True) as tape:
                z = np.random.normal(loc=0.0, scale=1.0, size=[flags.batch_size, flags.z_dim]).astype(np.float32)

                d_logits = D([G(z), z])
                d2_logits = D([batch_images, E(batch_images)])

                d_loss_real = tl.cost.sigmoid_cross_entropy(d2_logits, tf.ones_like(d2_logits), name='dreal')
                d_loss_fake = tl.cost.sigmoid_cross_entropy(d_logits, tf.zeros_like(d_logits), name='dfake')
                d_loss = d_loss_fake + d_loss_real

                g_loss = tl.cost.sigmoid_cross_entropy(d_logits, tf.ones_like(d_logits), name='gfake')

                e_loss = tl.cost.sigmoid_cross_entropy(d2_logits, tf.zeros_like(d2_logits), name='ereal')

            grad = tape.gradient(g_loss, G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            grad = tape.gradient(d_loss, D.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
            grad = tape.gradient(e_loss, E.trainable_weights)
            e_optimizer.apply_gradients(zip(grad, E.trainable_weights))

            del tape

            print("Epoch: [{}/{}] [{}/{}] took: {:.3f}, d_loss: {:.5f}, g_loss: {:.5f}, e_loss: {:.5f}".
                  format(epoch, flags.n_epoch, step, n_step_epoch, time.time() - step_time, d_loss, g_loss, e_loss))

        if np.mod(epoch, flags.save_every_epoch) == 0:
            G.save_weights('{}/G.npz'.format(flags.checkpoint_dir), format='npz')
            D.save_weights('{}/D.npz'.format(flags.checkpoint_dir), format='npz')
            E.save_weights('{}/E.npz'.format(flags.checkpoint_dir), format='npz')
            G.eval()
            result = G(z)
            G.train()
            tl.visualize.save_images(result.numpy(), [num_tiles, num_tiles],
                                     '{}/train_{:02d}.png'.format(flags.sample_dir, epoch))

            for step, batch_images in enumerate(images):
                if batch_images.shape[0] != flags.batch_size:
                    break
                result = G(E(batch_images))
                tl.visualize.save_images(batch_images.numpy(), [num_tiles, num_tiles],
                                         '{}/real_{:02d}.png'.format(flags.pair_dir, epoch))
                tl.visualize.save_images(result.numpy(), [num_tiles, num_tiles],
                                         '{}/reproduced_{:02d}.png'.format(flags.pair_dir, epoch))
                break


if __name__ == '__main__':
    train()
