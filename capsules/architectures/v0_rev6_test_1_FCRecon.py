#from operations.capsule_operations import *
import sys
sys.path.insert(0, 'capsules/operations')
from routing import *
from capsule_operations import *

import tensorflow as tf
import numpy as np
import collections

# CNN with a few layers, but notably different from 0v_rev4_test_1 because, a single depthwise_convolutional_capsule_layer layer is used
# to test that it is able to actually produce a reconstruction

# Use keras.layers to contract instead to see if it makes a difference

# CROP AND DOWNSAMPLE images to 64x64

Results = collections.namedtuple('Results', ('output'))

class architecture(object):
    def __init__(self):
        self.hparams = tf.contrib.training.HParams(
          decay_rate=0.9,
          decay_steps=3,
          learning_rate=1.e-5, # 0.001
          loss_type='margin',
          padding='VALID',
          remake=True,
          verbose=False,
          maximum_learning_rate = 1.e-30, # 1.e-7
      )


    def build(self, input_images):
        #input images = [batch, height,width]
        its = input_images.get_shape().as_list()
        '''input_images = tf.expand_dims(input_images, axis=3)
        print(input_images.get_shape().as_list())
        input_images = tf.image.crop_and_resize(input_images, [[0.25,0.25, 0.75,0.75]], [0], [64,64] )
        input_images = tf.squeeze(input_images, axis=3)'''
        print("Image size:")
        print(input_images.get_shape().as_list())
        #input_images = tf.expand_dims(input_images, axis=3)
        #input_images = tf.image.resize_images(input_images, size=[512,512])
        #input_images = tf.squeeze(input_images, axis=3)
        #input_images = tf.Print(input_images, input_images.get_shape().as_list(), "Memory check -1")
        print(">>>>> Initial Convolution")
        print(input_images.get_shape().as_list())
        layer1 = init_conv_2d(input_images, 512, "Init", kernel_size=21, type="SAME")# [batch, 1, num_ch, height, width]
        layer1 = tf.nn.relu(layer1, name="initrelu")


        tmp=layer1.get_shape().as_list()
        layer1b = tf.reshape(layer1, [tmp[0]] + [8, 64] + tmp[3:5])

        for i in range(1):
            layer1b = depthwise_convolutional_capsule_layer(layer1b, 15,15, "DepthConvInit"+str(i), strides=[1,1], num_output_channels=64, upsampling_factor=None, type="SAME", conv_vector=True)
            layer1b = tf.nn.relu(layer1b)
        for i in range(2):
            layer1b = depthwise_convolutional_capsule_layer(layer1b, 9,9, "DepthConv"+str(i), strides=[1,1], num_output_channels=64, upsampling_factor=None, type="SAME", conv_vector=True)
            layer1b = tf.nn.relu(layer1b)
        layer1b = depthwise_convolutional_capsule_layer(layer1b, 3,3, "DepthConvPre", strides=[1,1], num_output_channels=8, upsampling_factor=None, type="SAME", conv_vector=True)
        layer1b = tf.nn.relu(layer1b)



        '''layer1b = convolutional_capsule_layer_v2(layer1b, 9,9,'ConvCaps1',output_kernel_vec_dim=8,  num_output_channels=4, strides=[1,1], type="SAME")
        layer1b = convolutional_capsule_layer_v2(layer1b, 9,9,'ConvCaps2',output_kernel_vec_dim=8,  num_output_channels=8, strides=[2,2], type="SAME")
        layer1b = convolutional_capsule_layer_v2(layer1b, 3,3,'ConvCaps3',output_kernel_vec_dim=4,  num_output_channels=16, strides=[1,1], type="SAME")'''
        '''layer1b = convolutional_capsule_layer_v2(layer1b, 9,9,'ConvCaps1',output_kernel_vec_dim=8,  num_output_channels=4, strides=[2,2], type="SAME")
        layer1b = convolutional_capsule_layer_v2(layer1b, 9,9,'ConvCaps2',output_kernel_vec_dim=8,  num_output_channels=4, strides=[2,2], type="SAME")'''
        '''layer1b = depthwise_convolutional_capsule_layer(layer1b, 9,9, "DepthConvPre2", strides=[4,4], num_output_channels=4, upsampling_factor=None, type="SAME", conv_vector=True)'''

        layer1b = tf.reshape(layer1b, [its[0], -1])
        layer1b = tf.layers.dense(layer1b, 2048, activation=tf.nn.relu)
        layer1b = tf.layers.dense(layer1b, 1024, activation=tf.nn.relu)
        layer1b = tf.layers.dense(layer1b, 1024, activation=tf.nn.relu)
        layer1b = tf.layers.dense(layer1b, its[1]*its[2], activation=tf.sigmoid)
        output = tf.reshape(layer1b, its)


        '''layer1c = tf.transpose(layer1b, [0,3,4,1,2])
        tmp = layer1c.get_shape().as_list()
        layer1c = tf.reshape(layer1c, tmp[0:3] + [tmp[3]*tmp[4], 1])
        layer15 = tf.squeeze(layer1c, axis=4)
        layer20 = tf.layers.conv2d(layer15, 1, 1, padding='same', name="End5")
        layer21 = tf.nn.relu(layer20, name="Final")
        output = tf.squeeze(layer21, axis=[-1])'''

        '''for i in range(1):
            layer1b = depthwise_convolutional_capsule_layer(layer1b, 3,3, "DepthConvPost"+str(i), strides=[1,1], num_output_channels=8, upsampling_factor=None, type="SAME", conv_vector=True)
            layer1b = tf.nn.relu(layer1b)
        layer1b = depthwise_convolutional_capsule_layer(layer1b, 1,1, "DepthConvFinal", strides=[1,1], num_output_channels=1, upsampling_factor=None, type="SAME", conv_vector=True)
        layer1b = tf.nn.relu(layer1b)

        # reshape for use in keras
        layer1c = tf.transpose(layer1b, [0,3,4,1,2])
        tmp = layer1c.get_shape().as_list()
        layer1c = tf.reshape(layer1c, tmp[0:3] + [tmp[3]*tmp[4], 1])
        layer15 = tf.squeeze(layer1c, axis=4)'''

        '''for i in range(6,7):
            layer15 =tf.layers.conv2d(layer15, 512, 5, strides=(1,1), padding='same', name="red"+str(i))
            layer15 = tf.nn.relu(layer15)'''

        #layer18 = tf.layers.conv2d(layer15, 64, 3, padding='same', name='End3')
        #layer19=tf.nn.relu(layer18)
        '''layer20 = tf.layers.conv2d(layer15, 1, 1, padding='same', name="End5")
        layer21 = tf.nn.relu(layer20, name="Final")

        output = tf.squeeze(layer21, axis=[-1])'''



        #output = tf.add(output, input_images)
        print(output.get_shape().as_list())
        result = Results(output)
        print(">>>>> Graph Built!")

        with tf.device('/cpu:0'):
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False) # [] [unfinished]
            global_step = tf.add(global_step, 1)
        return result


    def loss_func(self, input_images, ground_truth):
        input_images = tf.expand_dims(input_images, axis=3)
        ground_truth = tf.expand_dims(ground_truth, axis=3)
        mini_batch_size = input_images.get_shape().as_list()[0]
        #input_images = tf.image.crop_and_resize(input_images, [[0.25,0.25, 0.75,0.75]]*mini_batch_size, list(range(mini_batch_size)), [64,64]  )
        #ground_truth = tf.image.crop_and_resize(ground_truth, [[0.25,0.25, 0.75,0.75]]*mini_batch_size, list(range(mini_batch_size)), [64,64]  )
        #input_images = tf.image.crop_and_resize(input_images, [[0.375,0.375,0.625,0.625]]*mini_batch_size, list(range(mini_batch_size)), [64,64]  )
        #ground_truth = tf.image.crop_and_resize(ground_truth, [[0.375,0.375,0.625,0.625]]*mini_batch_size, list(range(mini_batch_size)), [64,64]  )
        input_images = tf.image.crop_and_resize(input_images, [[0.375,0.375,0.625,0.625]]*mini_batch_size, list(range(mini_batch_size)), [32,32]  )
        ground_truth = tf.image.crop_and_resize(ground_truth, [[0.375,0.375,0.625,0.625]]*mini_batch_size, list(range(mini_batch_size)), [32,32]  )
        input_images = tf.squeeze(input_images, axis=3)
        ground_truth = tf.squeeze(ground_truth, axis=3)

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
        #print(">>>> New batch loss regulariser metric")
        #diff1 = tf.abs(tf.subtract(output, input_images))
        #zerovec1 = tf.zeros(shape=(1,1), dtype=tf.float32)
        #bool_mask = tf.not_equal(diff1, zerovec1)
        #diff1_omit = tf.boolean_mask(diff1, bool_mask)
        #diff1_omit = tf.reduce_mean(diff1_omit)
        #batch_loss = batch_loss + diff1_omit

        print(">>>> Find + and - loss")
        positive_loss =  tf.reduce_sum(tf.boolean_mask(difference, tf.greater(difference, 0.)))
        negative_loss =  tf.reduce_sum(tf.boolean_mask(difference, tf.less(difference, 0.)))

        print(">>>> PSNR and SSIM")
        psnr = tf.image.psnr(tf.expand_dims(ground_truth, axis=3), tf.expand_dims(output, axis=3), max_val=1114) #1114 3480
        ssim = tf.image.ssim(tf.expand_dims(ground_truth, axis=3), tf.expand_dims(output, axis=3), max_val=1114) #1114 3480

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
            model_output_f1 = tf.expand_dims(tf.slice(output, [0,0,0], [1, -1,-1]), axis=3)
            model_input_f1 = tf.expand_dims(tf.slice(input_images, [0,0,0], [1, -1,-1]), axis=3)
            model_gt_f1 = tf.expand_dims(tf.slice(ground_truth, [0,0,0], [1, -1,-1]), axis=3)
            tf.summary.image('model_output',  model_output_f1)
            tf.summary.image('model_input',  model_input_f1)
            tf.summary.image('model_ground_truth',  model_gt_f1)
            tf.summary.image('model_diff_gt_output',  model_gt_f1 - model_output_f1)
            tf.summary.image('model_diff_input_output',  model_input_f1 - model_output_f1)

        diagnostics = {'max_psnr': max_psnr, 'min_psnr': min_psnr, 'mean_psnr': mean_psnr, 'max_ssim':max_ssim, 'min_ssim':min_ssim, 'mean_ssim':mean_ssim, 'positive_loss':positive_loss, 'negative_loss':negative_loss, 'total_loss':total_loss}
        return output, batch_loss, diagnostics
