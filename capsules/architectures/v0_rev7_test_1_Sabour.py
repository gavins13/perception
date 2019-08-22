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
          decay_steps=3.,
          learning_rate=1.e-7, # 0.001
          loss_type='margin',
          padding='VALID',
          remake=True,
          verbose=False,
          maximum_learning_rate = 1.e-55, # 1.e-7
        )
        self.method = 'routing'

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
        layer1 = init_conv_2d(input_images, 512, "Init", kernel_size=13, type="SAME")# [batch, 1, num_ch, height, width]
        layer1 = tf.nn.relu(layer1, name="initrelu")

        tmp = layer1.get_shape().as_list()
        layer2 = tf.transpose(layer1, [0,3,4,1,2])
        layer2 = tf.reshape(layer2, [tmp[0],tmp[3],tmp[4]] + [-1, 1])
        layer2 = tf.squeeze(layer2, axis=4)
        layer2 = tf.layers.conv2d(layer2, 512, 9, padding='same', name='PrimaryCaps', activation=tf.nn.relu)
        layer2 = tf.expand_dims(layer2, axis=4)
        layer2 = tf.reshape(layer2, [tmp[0],tmp[3],tmp[4]] + [8, 64])
        layer2 = tf.transpose(layer2, [0,3,4,1,2])  # [batch, 8, num_ch, height, width]
        print("Layer2 shape")
        print(layer2.get_shape().as_list())
        if(self.method=='patch_routing'):
            layer3 = convolutional_capsule_layer_v2(layer2, 32,32,'ConvCaps1',output_kernel_vec_dim=16,  num_output_channels=10, strides=[1,1], type="VALID")
        elif(self.method=='routing'):
            with tf.variable_scope("matmul"):
                tmp = layer2.get_shape().as_list()
                matrix_shape = [1, tmp[2], tmp[3], tmp[4], 16, tmp[1]] # [1, num_ch, height, width, 16, vec_dim]
                matrix = variables.weight_variable(matrix_shape) # to keep [] to segment
                matrix = tf.tile(matrix, [tmp[0], 1, 1, 1, 1, 1]) # [batch, num_ch, height, width, 16, vec_dim]
                layer3 = tf.transpose(layer2, [0,2,3,4,1]) # [batch, num_ch, height, width, 8]
                layer3 = tf.expand_dims(layer3, axis=5) # [btach, num_ch, height, width, 8, 1]

                matrix_shape_bias = [1, tmp[2], 1,1,1,1] # [1, num_ch, 1,1, 1,1]
                with tf.variable_scope('matrix_bias'):
                    matrix_bias = variables.bias_variable(matrix_shape_bias)
                matrix_bias = tf.tile(matrix_bias, [tmp[0], 1, tmp[3], tmp[4], 16, 1])
                layer3 = tf.matmul(matrix, layer3) + matrix_bias  # [btach, num_ch, height, width, 16, 1]
                layer3 = tf.squeeze(layer3, axis=5)
                layer3 = tf.nn.relu(layer3)
                layer3 = tf.transpose(layer3, [0,4,1,2,3])
            layer3 = routing(layer3, 'Routing', output_dimensions=[20,1,1], squash_biases=None, num_routing=3, bias_channel_sharing=False)
        else:
            raise NotImplementedError()
        # [batch, 16, 10, 1, 1]
        print("Digitrouting")
        print(layer3.get_shape().as_list())

        layer1b = tf.reshape(layer3, [its[0], -1])
        layer1b = tf.layers.dense(layer1b, 768, activation=tf.nn.relu)
        layer1b = tf.layers.dense(layer1b, 2048, activation=tf.nn.relu)
        layer1b = tf.layers.dense(layer1b, 1024, activation=tf.nn.relu)
        layer1b = tf.layers.dense(layer1b, its[1]*its[2], activation=tf.sigmoid)
        layer1b = tf.reshape(layer1b, its)
        layer1b = tf.expand_dims(layer1b, axis=1)
        layer1b = tf.expand_dims(layer1b, axis=1)
        print("FC")
        print(layer1b.get_shape().as_list())

        layer1b = tf.norm(layer1b, axis=1)
        #layer1b = tf.multiply(layer1b, variables.bias_variable([1, tmp[2], tmp[3], tmp[4]]))
        #layer1b = tf.reduce_sum(layer1b, axis=1)
        #output = tf.reshape(layer1b, its)


        output = tf.transpose(layer1b, [0,2,3,1])
        for i in range(4):
            output =tf.layers.conv2d(output, 64, 9, strides=(1,1), padding='same', name="ending"+str(i), activation=tf.nn.relu)
        output = tf.layers.conv2d(output, 1, 1, strides=(1,1), padding='same', name="endingfinal", activation=tf.nn.relu)
        output = tf.squeeze(output, axis=3)

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
