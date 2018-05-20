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
        print(">>>>> Initial Convolution")
        print(np.shape(input_images))
        layer1 = init_conv_2d(input_images, 256, "Init", kernel_size=9)
        print(layer1.get_shape().as_list())

        print(">>>>> Initial Convolution Capsule")
        layer2 = convolutional_capsule_layer(layer1, 3,3,'ConvCaps',output_kernel_vec_dim=3,    num_output_channels=2, strides=[3,3])

        # layer2b = _routing(layer2, layer_dimensions=[batch, vec_dim, num_channels, height, width], squash_biases, num_routing)
        #print(layer2.get_shape().as_list())
        #print(">>>>> Routing 1")
        #layer2b = patch_based_routing(layer1,"primaryrouting", patch_shape=[1,5,5])
        #print(layer2b.get_shape().as_list())

        #print(">>>>> FC Capsule Layer")
        #layer3 = fc_capsule_layer(layer2, "Layer3", apply_weights=True,    share_weights_within_channel=False, output_vec_dimension=16)

        print(">>>>> Routing 2")
        layer3b = routing(layer2, "digitrouting", output_dimensions=[10,1,1],    bias_channel_sharing=True) # [batch, 16, num_ch, height, width] is equal to  [b,16,10,1,1]
        #layer3b = _patch_based_routing(layer3, patch_shape=None)  32 channel output
        #layer4 = logistic_fc_neural_network(flatten(layer3b))
        print(">>>>> Output")
        with tf.name_scope('output'):
            layer4 = tf.squeeze(layer3b, axis=[3,4])
            layer4 = tf.transpose(layer4, [0,2,1])
            output = tf.norm(layer4, axis=2) #out: [batch, 10]
        result = Results(output)

        print(">>>>> Graph Built!")
        return result
