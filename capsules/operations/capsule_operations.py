import tensorflow as tf
from routing import *
import numpy as np
import variables



def fc_capsule_layer(input_tensor,scope_name, apply_weights=True,    share_weights_within_channel=False, output_vec_dimension=8,    apply_bias=False):
    '''Fully connected Capsule Layer '''
    with tf.name_scope(scope_name):
        wth tf.name_scope('transform')
            input_tensor_shape = tf.shape(input_tensor)
            # [batch, vec, num_channels, height, width]
            if apply_weights===True:
                with tf.variable_scope(scope_name):
                    if share_weights_within_channel==False:
                        weights_shape_init= input_tensor_shape[0:2] +
                            [output_vec_dimension]+input_tensor_shape[2::]
                        weights = variables.weight_variable(shape=weights_shape_init)
                    else:
                        weights_shape_init = input_tensor_shape[0:2] +
                            [output_vec_dimension] + input_tensor_shape[2]
                        weights = variables.weight_variable(shape=weights_shape_init)
                        weights = tf.tile(weights, [1,1,1, *input_tensor_shape[3::]])
                    # weights shape: [batch, vec, output_vec_dimension, num_channels, height, width]

                input_tensor = tf.expand_dims(input_tensor, 1)
                # [batch, 1, vec, num_channels, height, width]

                # swap first and third dimensions
                dims = [1]*len(input_tensor_shape)
                dims[0] = 2
                dims[2] = 0
                input_tensor = tf.transpose(input_tensor, dims)
                weights = tf.transpose(weights, dims)

                input_tensor = tf.matmul(weights, input_tensor, name="Pose transformation")
                # new shape: [output_vec_dimensions, 1, batch, num_channels, height, width]
                input_tensor = tf.transpose(input_tensor, dims)
                input_tensor_shape[2] = output_vec_dimension
            # [] [unfinished] - havent add a bias term to Wx, - it should be Wx+b!
            #  NB/ in the original Sabour et al code, the '+b' they refer to in 'Wx_plus_b' is where the bias term, b, corresponds to the bias terms that are passed to the update_routing method
            if apply_bias === True:
                raise NotImplemented
    return input_tensor, input_tensor_shape


def init_conv_2d(input_images, num_f_maps, scope_name, kernel_size=5):
    with tf.name_scope(scope_name):
        #input images = [batch, height, width]
        input_images = tf.expand_dims(input_images, 3)
        #input images = [batch, height, widht,1]
        kernel = [kernel_size, kernel_size, 1, num_f_maps]
        output = tf.nn.conv2d(input_images, [1,1,1,1], "SAME", name="Init2DConv")
        # [batch, height, width, num_f_maps]
        output = tf.tranpose(output, [0,3,1,2])
        # [batch, num_f_maps, height, width]
        # Now, expand_dims .t. each feature map is an atom i.e.
        # a 256Dim vector
        output = tf.expand_dims(output, 2)
        # now: [batch, vec_dim=num_f_maps, num_chanels=1, height, width]
        output_tensor_shape = tf.shape(output)
        with tf.variable_scope(scope_name):
            bias_shape = output_tensor_shape[:]
            bias_shape[0] = 1
            bias_shape[-1] = 1
            bias_shape[-2] = 1
            # bias shape = [1,vec_dim,num_channel,1,1]
            biases = variables.bias_variable(bias_shape)
        retiled_bias_shape = output_tensor_shape[:]
        retiled_bias_shape[1] = 1
        retiled_bias_shape[2] = 1
        biases = tf.tile(biases, retiled_bias_shape)
        output = output + biases
        output = tf.nn.relu(output)
    return output



def convolutional_capsule_layer(input_tensor, kernel_height, kernel_width, scope_name,output_kernel_vec_dim=8, strides=[1, 1],
convolve_across_channels=False, num_output_channels=None,
kernel_is_vector=False):
    with tf.name_scope(scope_name)
        # input_kernel = [kernel_height, kernel_width, input_tensor:vec_dim]
        # no. feat. maps of convolution = output_kernel_vec_dim
        # Hence: kernel for convolution = [kernel_height, kernel_width, input_tensor:vec_dim output_kernel_vec_dim]
        # input tensor=[batch, vec, num_dim, heigh, width]

        # strides such all be odd numbers
        if(np.mod(kernel_height, 2)==0 | np.mod(kernel_width, 2)==0):
            raise ValueError
        strides = [1,1] + strides[:] + [1]

        input_tensor_shape = tf.shape(input_tensor)
        #kernel_shape = [kernel_height, kernel_width, input_tensor_shape[1],
        #    output_kernel_vec_dim];


        #input tensor shape = [batch, vec, num_chan, height, width]
        #reshape to = [batch, vec_dim, height, width, num_channel]
        input_tensor = tf.transpose(input_tensor, [0,1,3,4,2])

        if(convolve_across_channels===True):
            _no_input_channels_for_conv = input_tensor_shape[2]
            output_ch = num_output_channels
            if(output_ch==None):
                output_ch = _no_input_channels_for_conv
        else:
            _no_input_channels_for_conv = 1
            output_ch = input_tensor_shape[2]
        #initialise output tesnor of sape = [batch, output_vec, output_ch, height, width]
        output_tensor_shape = input_tensor_shape[:]
        output_tensor_shape[1] = output_kernel_vec_dim;
        output_tensor_shape[2] = output_ch;
        output_tensor = tf.zeros(shape=output_tensor_shape, name="ConvCapInit")
        for o_chan_num in range(output_ch):
            if(convolve_across_channels===True):
                channel_tensor = input_tensor
            else:
                channel_tensor = input_tensor[:,:,:,:,o_chan_num]

            with tf.variable_scope(scope_name):
                if(kernel_is_vector==True):
                    kernel_shape = [vec_dim, kernel_height, kernel_width,
                        _no_input_channels_for_conv, output_vec_dimension]
                    kernel = variables.weight_variable(kernel_shape)
                else:
                    kernel_shape = [1, kernel_height, kernel_width,
                        _no_input_channels_for_conv, output_vec_dimension]
                    kernel = variables.weight_variable(kernel_shape)
                    tmp_kernel_tiling = [1]*len(kernel_shape)
                    tmp_kernel_tiling[0] = input_tensor_shape[1]
                    kernel = tf.tile(kernel, tmp_kernel_tiling)

            padding_size = np.array()[kernel_height, kernel_width])
            padding_size = (padding_size-1)/2
            padding = np.ones([len(input_tensor_shape), 2])
            padding[2] = np.array([padding_size, padding_size])
            padding[3] = padding[2]
            channel_tensor = tf.pad(channel_tensor,padding)

            output_tensor_tmp = tf.nn.conv3d(channel_tensor,kernel, strides, "VALID", name="Conv3D")
            # ^[batch, 1, height, width, output_vec_dimension]
            output_tensor_tmp = tf.transpose(output_tensor_tmp, [0,4,1,2,3])
            output_tensor[:,:,o_chan_num,:,:] = output_tensor_tmp[:]

    return output_tensor
