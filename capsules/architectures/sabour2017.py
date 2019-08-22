#from operations.capsule_operations import *
import sys
sys.path.insert(0, 'capsules/operations')
from routing import *
from capsule_operations import *

import tensorflow as tf
import numpy as np
import collections


Results = collections.namedtuple('Results', ('output'))

class architecture(object):
    def __init__(self):
        self.hparams = tf.contrib.training.HParams(
          decay_rate=0.96,
          decay_steps=2000,
          learning_rate=0.001,
          loss_type='margin',
          padding='VALID',
          remake=True,
          verbose=False,
      )

    def loss_func(self):
        raise NotImplementedError()

    def build(self, input_images):
        #input images = [batch, height,width]

        #input_images = tf.expand_dims(input_images, axis=3)
        #input_images = tf.image.resize_images(input_images, size=[512,512])
        #input_images = tf.squeeze(input_images, axis=3)
        #input_images = tf.Print(input_images, input_images.get_shape().as_list(), "Memory check -1")
        print(">>>>> Initial Convolution")
        print(np.shape(input_images))
        layer1 = init_conv_2d(input_images, 256, "Init", kernel_size=9, type="VALID")# [batch, 1, num_ch, height, width]
        layer1 = tf.Print(layer1, layer1.get_shape().as_list(), "Memory check 0")
        tmp=layer1.get_shape().as_list()

        layer2 = tf.reshape(layer1, [tmp[0]] + [8, 32] + tmp[3:5])

        k_size=tmp[3]
        print(">>>>> Initial Convolution Capsule")
        layer2 = convolutional_capsule_layer(layer2, k_size,k_size,'ConvCaps',output_kernel_vec_dim=16, num_output_channels=10, strides=[1,1], type="VALID")
        layer2 = tf.Print(layer2, layer2.get_shape().as_list(), "Memory check 1")
        print("CONV CAPS SHAPE")
        print(layer2.get_shape().as_list())
        '''print(">>>>> Initial Convolution Capsule 2")
        layer2 = convolutional_capsule_layer(layer2, k_size,k_size,'ConvCaps2',output_kernel_vec_dim=16, num_output_channels=1, strides=[1,1])
        layer2 = tf.Print(layer2, layer2.get_shape().as_list(), "Memory check 2")
        print("CONV CAPS SHAPE 2")
        print(layer2.get_shape().as_list())
        print(">>>>> Initial Convolution Capsule 3")
        layer2 = convolutional_capsule_layer(layer2, k_size,k_size,'ConvCaps3',output_kernel_vec_dim=128, num_output_channels=1, strides=[1,1])
        layer2 = tf.Print(layer2, layer2.get_shape().as_list(), "Memory check 3")'''

        #print(">>>>> Routing 2")
        #layer3b = routing(layer2, "digitrouting", output_dimensions=[10,1,1],    bias_channel_sharing=True) # [batch, 16, num_ch, height, width] is equal to  [b,16,10,1,1]
        #layer3b = _patch_based_routing(layer3, patch_shape=None)  32 channel output
        #layer4 = logistic_fc_neural_network(flatten(layer3b))
        print(">>>>> Output")
        with tf.name_scope('output'):
            layer4 = tf.squeeze(layer2, axis=[3,4])
            layer4 = tf.transpose(layer4, [0,2,1])
            output = tf.norm(layer4, axis=2) #out: [batch, 10]
        result = Results(output)

        print(">>>>> Graph Built!")
        return result
