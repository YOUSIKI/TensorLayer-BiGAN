import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (
    Input, Dense, DeConv2d, BatchNorm2d, Conv2d,
    Reshape, Flatten, Dropout, Concat
)


w_init = tf.random_normal_initializer(stddev=0.02)
g_init = tf.random_normal_initializer(1.0, 0.02)
lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)


def get_encoder(x_shape, z_shape, ef_dim=64):
    ni = Input(x_shape)
    nn = Conv2d(ef_dim, (5, 5), (2, 2), act=lrelu, W_init=w_init)(ni)
    nn = Conv2d(ef_dim * 2, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(nn)
    nn = Conv2d(ef_dim * 4, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(nn)
    nn = Conv2d(ef_dim * 8, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(nn)
    nn = Flatten()(nn)
    nn = Dense(n_units=np.prod(z_shape[1:]), act=tf.identity, W_init=w_init)(nn)
    nn = Reshape([i if i else -1 for i in z_shape])(nn)

    return tl.models.Model(inputs=ni, outputs=nn, name='encoder')


def get_generator(x_shape, z_shape, gf_dim=64):
    div16 = x_shape[1] // 16

    ni = Input(z_shape)
    nn = Dense(n_units=(gf_dim * 8 * div16 * div16), W_init=w_init, b_init=None)(ni)
    nn = Reshape(shape=[-1, div16, div16, gf_dim * 8])(nn)
    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=g_init,name=None)(nn)
    nn = DeConv2d(gf_dim * 4, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=g_init)(nn)
    nn = DeConv2d(gf_dim * 2, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=g_init)(nn)
    nn = DeConv2d(gf_dim, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=g_init)(nn)
    nn = DeConv2d(3, (5, 5), (2, 2), act=tf.nn.tanh, W_init=w_init)(nn)

    return tl.models.Model(inputs=ni, outputs=nn, name='generator')


def get_discriminator(x_shape, z_shape, df_dim=64):
    xi = Input(x_shape)
    zi = Input(z_shape)
    xn = Conv2d(df_dim, (5, 5), (2, 2), act=lrelu, W_init=w_init)(xi)
    xn = Conv2d(df_dim * 2, (5, 5), (2, 2), W_init=w_init, b_init=None)(xn)
    xn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(xn)
    xn = Dropout(keep=0.8)(xn)
    xn = Conv2d(df_dim * 4, (5, 5), (2, 2), W_init=w_init, b_init=None)(xn)
    xn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(xn)
    xn = Dropout(keep=0.8)(xn)
    xn = Conv2d(df_dim * 8, (5, 5), (2, 2), W_init=w_init, b_init=None)(xn)
    xn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(xn)
    xn = Dropout(keep=0.8)(xn)
    xn = Flatten()(xn)
    zn = Flatten()(zi)
    zn = Dense(n_units=df_dim * 8, act=lrelu)(zn)
    zn = Dropout(keep=0.8)(zn)
    nn = Concat()([xn, zn])
    nn = Dense(n_units=1, act=tf.identity, W_init=w_init)(nn)

    return tl.models.Model(inputs=[xi, zi], outputs=nn, name='discriminator')
