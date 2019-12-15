import numpy as np
import tensorflow as tf
import tensorlayer as tl


class FLAGS(object):
    def __init__(self):
        self.n_epoch = 25 # "Epoch to train [25]"
        self.z_dim = 128 # "Num of noise value [128]"
        self.lr = 0.0002 # "Learning rate of for adam [0.0002]"
        self.beta1 = 0.5 # "Momentum term of adam [0.5]"
        self.batch_size = 64 # "The number of batch images [64]"
        self.output_size = 64 # "The size of the output images to produce [64]"
        self.sample_size = 64 # "The number of sample images [64]"
        self.c_dim = 3 # "Number of image channels. [3]"
        self.save_every_epoch = 1 # "The interval of saveing checkpoints."
        self.load_weights = None # "weights to load before training and evaluation [None]"
        self.checkpoint_dir = "checkpoints" # "Directory name to save the checkpoints [checkpoints]"
        self.sample_dir = "samples" # "Directory name to save the image samples [samples]"


flags = FLAGS()

tl.logging.set_verbosity(tl.logging.DEBUG)  ## enable debug logging

tl.files.exists_or_mkdir(flags.checkpoint_dir) # save model
tl.files.exists_or_mkdir(flags.sample_dir) # save generated image