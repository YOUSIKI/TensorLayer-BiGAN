import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (Input, Dense, DeConv2d, Reshape, BatchNorm2d, Conv2d, Flatten, Dropout, Concat)
from data import flags
import numpy as np

def get_generator(shape, gf_dim=64): # Dimension of gen filters in first conv layer. [64]
    image_size = 64
    s16 = image_size // 16

    w_init = tf.random_normal_initializer(0.0, 0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    ni = Input(shape)
    nn = Dense(n_units=(gf_dim * 8 * s16 * s16), W_init=w_init, b_init=None)(ni)
    nn = Reshape(shape=[-1, s16, s16, gf_dim*8])(nn)
    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)
    nn = DeConv2d(gf_dim * 4, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(nn)
    nn = DeConv2d(gf_dim * 2, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(nn)
    nn = DeConv2d(gf_dim, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(nn)
    nn = DeConv2d(3, (5, 5), (2, 2), act=tf.nn.tanh, W_init=w_init)(nn)

    return tl.models.Model(inputs=ni, outputs=nn, name='generator')

def get_encoder(shape, ef_dim=64):
    w_init = tf.random_normal_initializer(0.0, 0.02)
    gamma_init = tf.random_normal_initializer(1.0, 0.02)
    lrelu = lambda x : tf.nn.leaky_relu(x, 0.2)

    ni = Input(shape)
    nn = Conv2d(ef_dim, (5,5), (2,2), act=lrelu, W_init=w_init)(ni)
    nn = Conv2d(ef_dim*2, (5,5), (2,2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(nn)
    nn = Conv2d(ef_dim*4, (5,5), (2,2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(nn)
    nn = Conv2d(ef_dim*8, (5,5), (2,2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(nn)
    nn = Flatten()(nn)
    nn = Dense(n_units=flags.z_dim, W_init=w_init, b_init=None)(nn)
    return tl.models.Model(inputs=ni, outputs=nn, name='encoder')


def get_discriminator(latent_shape, image_shape, df_dim=64):

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)

    n1i = Input(image_shape)
    n1 = Conv2d(df_dim, (5, 5), (2, 2), act=lrelu, W_init=w_init)(n1i)
    n1 = Conv2d(df_dim*2, (5, 5), (2, 2), W_init=w_init, b_init=None)(n1)
    n1 = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n1)
    n1 = Dropout(keep=0.8)(n1)
    n1 = Conv2d(df_dim*4, (5, 5), (2, 2), W_init=w_init, b_init=None)(n1)
    n1 = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n1)
    n1 = Dropout(keep=0.8)(n1)
    n1 = Conv2d(df_dim*8, (5, 5), (2, 2), W_init=w_init, b_init=None)(n1)
    n1 = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n1)
    n1 = Dropout(keep=0.8)(n1)
    n1 = Flatten()(n1)  # [-1,4*4*df_dim*8]

    n2i = Input(latent_shape)
    n2 = Dense(n_units=4*4*df_dim*8, W_init=w_init, b_init=None)(n2i)
    n2 = Dropout(keep=0.8)(n2)
    nn = Concat()([n1, n2])

    nn = Dense(n_units=1, W_init=w_init, b_init=None)(nn)

    return tl.models.Model(inputs=[n1i, n2i], outputs=nn, name='discriminator')

