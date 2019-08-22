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

    def build(self, input_images):
        #input images = [batch, height,width]

        input_images = tf.expand_dims(input_images, axis=3)
        input_images = tf.image.resize_images(input_images, size=[128,128])
        input_images = tf.squeeze(input_images, axis=3)
        input_images = tf.Print(input_images, input_images.get_shape().as_list(), "Memory check -1")
        print(">>>>> Initial Convolution")
        print(np.shape(input_images))
        layer1 = init_conv_2d(input_images, 16, "Init", kernel_size=9)# [batch, 1, num_ch, height, width]
        layer1 = tf.Print(layer1, layer1.get_shape().as_list(), "Memory check 0")
        tmp=layer1.get_shape().as_list()

        layer1 = tf.reshape(layer1, [tmp[0]] + [16, 1] + tmp[3:5])

        k_size=5
        print(">>>>> Initial Convolution Capsule")
        layer2 = convolutional_capsule_layer(layer1, k_size,k_size,'ConvCaps',output_kernel_vec_dim=16,    num_output_channels=2, strides=[2,2])
        layer2 = tf.Print(layer2, layer2.get_shape().as_list(), "Memory check 1")
        print("CONV CAPS SHAPE")
        print(layer2.get_shape().as_list())
        print(">>>>> Initial Convolution Capsule 2")
        layer2 = convolutional_capsule_layer(layer2, k_size,k_size,'ConvCaps2',output_kernel_vec_dim=16,    num_output_channels=4, strides=[1,1])
        layer2 = tf.Print(layer2, layer2.get_shape().as_list(), "Memory check 2")
        print("CONV CAPS SHAPE 2")
        print(layer2.get_shape().as_list())
        print(">>>>> Initial Convolution Capsule 3")
        layer2 = convolutional_capsule_layer(layer2, k_size,k_size,'ConvCaps3',output_kernel_vec_dim=32,    num_output_channels=4, strides=[2,2])
        layer2 = tf.Print(layer2, layer2.get_shape().as_list(), "Memory check 3")
        print("CONV CAPS SHAPE 3")
        print(layer2.get_shape().as_list())
        layer2 = convolutional_capsule_layer(layer2, k_size,k_size,'ConvCaps3a',output_kernel_vec_dim=32,    num_output_channels=8, strides=[1,1])
        layer2 = tf.Print(layer2, layer2.get_shape().as_list(), "Memory check 4")
        #print("CONV CAPS SHAPE 3a")
        #print(layer2.get_shape().as_list())
        #print(">>>>> Initial Convolution Capsule 3a")
        #layer2 = convolutional_capsule_layer(layer2, 1,1,'ConvCaps3b',output_kernel_vec_dim=32,    num_output_channels=8, strides=[1,1])
        #layer2 = tf.Print(layer2, layer2.get_shape().as_list(), "Memory check 5")
        print("CONV CAPS SHAPE 3b")
        print(layer2.get_shape().as_list())
        print(">>>>> Initial Convolution Capsule 4")
        layer2 = convolutional_capsule_layer(layer2, k_size,k_size,'ConvCaps4',output_kernel_vec_dim=32,    num_output_channels=4, strides=[1,1])
        layer2 = tf.Print(layer2, layer2.get_shape().as_list(), "Memory check 6")
        print("CONV CAPS SHAPE 4")
        print(layer2.get_shape().as_list())
        layer2 = convolutional_capsule_layer(layer2, k_size,k_size,'ConvCaps5',output_kernel_vec_dim=16,    num_output_channels=4, strides=[1,1], upsampling_factor=2)
        layer2 = tf.Print(layer2, layer2.get_shape().as_list(), "Memory check 7")
        print("CONV CAPS SHAPE 5")
        print(layer2.get_shape().as_list())
        layer2 = convolutional_capsule_layer(layer2, k_size,k_size,'ConvCaps6',output_kernel_vec_dim=16,    num_output_channels=4, strides=[1,1])
        layer2 = tf.Print(layer2, layer2.get_shape().as_list(), "Memory check 8")
        print("CONV CAPS SHAPE 6")
        print(layer2.get_shape().as_list())
        layer2 = convolutional_capsule_layer(layer2, k_size,k_size,'ConvCaps7',output_kernel_vec_dim=16,    num_output_channels=2, strides=[1,1], upsampling_factor=2)
        layer2 = tf.Print(layer2, layer2.get_shape().as_list(), "Memory check 9")
        print("CONV CAPS SHAPE 7")
        tmp=layer2.get_shape().as_list()
        k_size=tmp[3]
        layer2 = convolutional_capsule_layer(layer2, k_size,k_size,'ConvCaps8',output_kernel_vec_dim=16,    num_output_channels=10, strides=[1,1], type="VALID")
        layer2 = tf.Print(layer2, layer2.get_shape().as_list(), "Memory check 10")
        print("CONV CAPS SHAPE 8")
        print(layer2.get_shape().as_list())
        #print(">>>>> Initial Convolution Capsule 4")
        #layer2 = convolutional_capsule_layer(layer2, 3,3,'ConvCaps4',output_kernel_vec_dim=16,    num_output_channels=128, strides=[1,1])

        # layer2b = _routing(layer2, layer_dimensions=[batch, vec_dim, num_channels, height, width], squash_biases, num_routing)
        #print(layer2.get_shape().as_list())
        #print(">>>>> Routing 1")
        #layer2b = patch_based_routing(layer1,"primaryrouting", patch_shape=[1,5,5])
        #print(layer2b.get_shape().as_list())

        #print(">>>>> FC Capsule Layer")
        #layer3 = fc_capsule_layer(layer2, "Layer3", apply_weights=True,    share_weights_within_channel=False, output_vec_dimension=16)

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
