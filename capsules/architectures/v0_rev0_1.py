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

        layer2 = depthwise_convolutional_capsule_layer(layer1b, 9,9, "InitCaps_red", strides=[2,2], num_output_channels=2, upsampling_factor=None, type="SAME", conv_vector=True)


        layer3 = convolutional_capsule_layer_v2(layer2,5,5,'ConvCaps1',output_kernel_vec_dim=16,    num_output_channels=4, strides=[1,1], type="SAME")
        layer3 = tf.Print(layer3, layer2.get_shape().as_list(), "Memory check 00")


        layer4 = convolutional_capsule_layer_v2(layer3,5,5,'ConvCaps2_red',output_kernel_vec_dim=32,  num_output_channels=4, strides=[2,2], type="SAME")
        layer4 = tf.Print(layer4, layer2.get_shape().as_list(), "Memory check 00")


        layer5 = convolutional_capsule_layer_v2(layer4,5,5,'ConvCaps3',output_kernel_vec_dim=32,  num_output_channels=8, strides=[1,1], type="SAME")
        layer5 = tf.Print(layer5, layer2.get_shape().as_list(), "Memory check 00")

        ######################################################
        layer6 = convolutional_capsule_layer_v2(layer5,5,5,'ConvCaps4_red',output_kernel_vec_dim=64,  num_output_channels=8, strides=[2,2], type="SAME")
        layer6 = tf.Print(layer6, layer2.get_shape().as_list(), "Memory check 00")


        layer7 = convolutional_capsule_layer_v2(layer6,5,5,'ConvCaps5',output_kernel_vec_dim=32,  num_output_channels=8, strides=[1,1], type="SAME")
        layer7 = tf.Print(layer7, layer2.get_shape().as_list(), "Memory check 00")
        ###########################################################
        layer8 = depthwise_convolutional_capsule_layer(layer7, 4,4, "DecovCaps1", strides=[2,2], num_output_channels=8, upsampling_factor=2., type="SAME", conv_vector=False)
        print(layer5.get_shape().as_list())
        print(layer8.get_shape().as_list())
        layer8b = tf.concat([layer5,layer8], axis=2)
        print(layer8b.get_shape().as_list())

        layer9 = convolutional_capsule_layer_v2(layer8b,5,5,'ConvCaps6',output_kernel_vec_dim=16,  num_output_channels=8, strides=[1,1], type="SAME")

        layer10 = depthwise_convolutional_capsule_layer(layer9, 4,4, "DecovCaps2", strides=[2,2], num_output_channels=4, upsampling_factor=2., type="SAME", conv_vector=False)

        layer10b = tf.concat([layer10, layer3], axis=2)
        layer11 = convolutional_capsule_layer_v2(layer10b,5,5,'ConvCaps7',output_kernel_vec_dim=16,  num_output_channels=4, strides=[1,1], type="SAME")

        layer12 = depthwise_convolutional_capsule_layer(layer11, 4,4, "DecovCaps3", strides=[2,2], num_output_channels=2, upsampling_factor=2., type="SAME", conv_vector=False)

        print(layer12.get_shape().as_list())
        print(layer1b.get_shape().as_list())
        layer12a = tf.concat([layer12, layer1b], axis=2)
        layer12b = convolutional_capsule_layer_v2(layer12a, 5,5,'ConvCaps7a',output_kernel_vec_dim=16,  num_output_channels=2, strides=[1,1], type="SAME")


        layer13 = convolutional_capsule_layer_v2(layer12b, 5,5,'ConvCaps8',output_kernel_vec_dim=16,  num_output_channels=1, strides=[1,1], type="SAME")


        layer13a = convolutional_capsule_layer_v2(layer13, 5,5,'ConvCaps9',output_kernel_vec_dim=1,  num_output_channels=1, strides=[1,1], type="SAME")
        layer13b = tf.norm(layer13, axis=1, keepdims=True)
        layer14 = tf.concat([layer13a,layer13b], axis=2)
        layer15 = convolutional_capsule_layer_v2(layer14, 1,1,'ConvCaps10',output_kernel_vec_dim=1,  num_output_channels=1, strides=[1,1], type="SAME")


        output = tf.squeeze(layer15, axis=[1,2])
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
