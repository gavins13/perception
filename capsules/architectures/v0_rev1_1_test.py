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

        #input_images = tf.expand_dims(input_images, axis=3)
        #input_images = tf.image.resize_images(input_images, size=[512,512])
        #input_images = tf.squeeze(input_images, axis=3)
        #input_images = tf.Print(input_images, input_images.get_shape().as_list(), "Memory check -1")
        print(">>>>> Initial Convolution")
        print(input_images.get_shape().as_list())
        layer1 = init_conv_2d(input_images, 16, "Init", kernel_size=9, type="SAME")# [batch, 1, num_ch, height, width]
        layer1 = tf.Print(layer1, layer1.get_shape().as_list(), "Memory check 0")
        tmp=layer1.get_shape().as_list()



        layer1b = tf.reshape(layer1, [tmp[0]] + [16, 1] + tmp[3:5])

        layer2 = depthwise_convolutional_capsule_layer(layer1b, 25,25, "DepthConv1", strides=[1,1], num_output_channels=2, upsampling_factor=None, type="SAME", conv_vector=True)

        layer3 = depthwise_convolutional_capsule_layer(layer2, 9,9, "DepthConv2_red", strides=[2,2], num_output_channels=4, upsampling_factor=None, type="SAME", conv_vector=True)

        layer4 = depthwise_convolutional_capsule_layer(layer3, 9,9, "DepthConv3", strides=[1,1], num_output_channels=4, upsampling_factor=None, type="SAME", conv_vector=True)

        layer5 = depthwise_convolutional_capsule_layer(layer4, 5,5, "DepthConv4_red", strides=[2,2], num_output_channels=8, upsampling_factor=None, type="SAME", conv_vector=True)

        layer6 = depthwise_convolutional_capsule_layer(layer5, 5,5, "DepthConv5_red", strides=[2,2], num_output_channels=8, upsampling_factor=None, type="SAME", conv_vector=True)

        layer7 = convolutional_capsule_layer_v2(layer6, 3,3,'ConvCaps1',output_kernel_vec_dim=16,  num_output_channels=2, strides=[1,1], type="SAME")

        layer9 = depthwise_convolutional_capsule_layer(layer7, 5,5, "DepthConv6_up", strides=[2,2], num_output_channels=8, upsampling_factor=2., type="SAME", conv_vector=False)

        layer10 = tf.concat([layer9, layer5], axis=2)

        layer11 = depthwise_convolutional_capsule_layer(layer10, 5,5, "DepthConv7_up", strides=[2,2], num_output_channels=4, upsampling_factor=2., type="SAME", conv_vector=False)

        layer12 = depthwise_convolutional_capsule_layer(layer11, 9,9, "DepthConv8_up", strides=[2,2], num_output_channels=2, upsampling_factor=2., type="SAME", conv_vector=False)

        layer13 = depthwise_convolutional_capsule_layer(layer12, 25,25, "DepthConv9", strides=[1,1], num_output_channels=1, upsampling_factor=1., type="SAME", conv_vector=False)

        layer13 = tf.norm(layer13, axis=1, keepdims=True)

        layer14 = tf.nn.relu(layer13)

        output = tf.squeeze(layer14, axis=[1,2])
        print(output.get_shape().as_list())
        result = Results(output)
        print(">>>>> Graph Built!")
        return result


    def loss_func(self, input_images, ground_truth):

        print(">>>Start Building Architecture.")
        res = self.build(input_images)
        print(">>>Finished Building Architecture.")
        output = res.output

        print(">>>Some Maths on result")
        print(">>>> Find Difference")
        difference = tf.subtract(ground_truth, output)
        print(">>>> Find Norm")
        L2_norm = tf.norm(difference, axis=[1,2])
        print(">>>> Find Mean of Norm")
        batch_loss = tf.reduce_mean(L2_norm)

        print(">>>> Find + and - loss")
        positive_loss =  tf.reduce_sum(tf.boolean_mask(difference, tf.greater(difference, 0.)))
        negative_loss =  tf.reduce_sum(tf.boolean_mask(difference, tf.less(difference, 0.)))

        print(">>>> PSNR and SSIM")
        psnr = tf.image.psnr(tf.expand_dims(ground_truth, axis=3), tf.expand_dims(output, axis=3), max_val=65535)
        ssim = tf.image.ssim(tf.expand_dims(ground_truth, axis=3), tf.expand_dims(output, axis=3), max_val=65535)

        print(">>>> PSNR Stats")
        max_psnr = tf.reduce_max(psnr)
        min_psnr = tf.reduce_min(psnr)
        mean_psnr = tf.reduce_mean(psnr)


        print(">>>> SSIM Stats")
        max_ssim = tf.reduce_max(ssim)
        min_ssim = tf.reduce_min(ssim)
        mean_ssim = tf.reduce_mean(ssim)


        print(">>>> Find Mean Loss")
        with tf.name_scope('total'):
            print(">>>>>> Add to collection")
            tf.add_to_collection('losses', batch_loss)
            print(">>>>>> Creating summary")
            tf.summary.scalar(name='batch_L2_reconstruction_cost', tensor=batch_loss)
            print(">>>> Add result to collection of loss results for this tower")
            all_losses = tf.get_collection('losses') # [] , this_tower_scope) # list of tensors returned
            total_loss = tf.add_n(all_losses) # element-wise addition of the list of tensors
            #print(total_loss.get_shape().as_list())
            tf.summary.scalar('total_loss', total_loss)
        print(">>>> Add results to output")
        with tf.name_scope('accuracy'):
            tf.summary.scalar('max_psnr', max_psnr)
            tf.summary.scalar('min_psnr', min_psnr)
            tf.summary.scalar('mean_psnr', mean_psnr)
            tf.summary.scalar('max_ssim', max_ssim)
            tf.summary.scalar('min_ssim', min_ssim)
            tf.summary.scalar('mean_ssim', mean_ssim)
            tf.summary.scalar('positive_loss', positive_loss)
            tf.summary.scalar('negative_loss', tf.multiply(negative_loss, -1.))
        return output, batch_loss
