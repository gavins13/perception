import tensorflow as tf
from routing import *
import numpy as np
import variables



def fc_capsule_layer(input_tensor,scope_name, apply_weights=True,    share_weights_within_channel=False, output_vec_dimension=8,    apply_bias=False):
    '''Fully connected Capsule Layer '''
    with tf.name_scope(scope_name):
        with tf.name_scope('transform'):
            print(">>>>>>> Initialise weights")
            print(input_tensor.get_shape().as_list())
            input_tensor_shape = input_tensor.get_shape().as_list()
            # [batch, vec, num_channels, height, width]
            if(apply_weights==True):
                with tf.variable_scope(scope_name):
                    if share_weights_within_channel==False:
                        weights_shape_init= input_tensor_shape[0:2] +       [output_vec_dimension]+input_tensor_shape[2::]
                        weights = variables.weight_variable(shape=weights_shape_init)
                    else:
                        weights_shape_init = input_tensor_shape[0:2] +                            [output_vec_dimension] + input_tensor_shape[2]
                        weights = variables.weight_variable(shape=weights_shape_init)
                        weights = tf.tile(weights, [1,1,1, *input_tensor_shape[3::]])
                    # weights shape: [batch, vec, output_vec_dimension, num_channels, height, width]

                print(">>>>>>> Setup matrix multiplication")
                input_tensor = tf.expand_dims(input_tensor, 1)
                # [batch, 1, vec, num_channels, height, width]

                # swap first and third dimensions
                dims = list(range(len(input_tensor_shape)+1))
                dims[0] = 2
                dims[2] = 0
                print(">>>>>>>>> Transpose input")
                print(input_tensor.get_shape().as_list())
                print(dims)
                input_tensor = tf.transpose(input_tensor, dims) # [vec, 1, batch, num_ch, height, width]
                print(">>>>>>>>> Transpose weights")
                print(weights.get_shape().as_list())
                print(dims)
                weights = tf.transpose(weights, dims) # [output_vec_dimsion, vec, batch, num_ch, height, width]
                print(">>>>>>> Multiply with weight matrix")
                print(">>>>>>>>>>> Transpose for multiplication")
                dims = list(range(len(input_tensor_shape)+1))
                dims = list(range(2,len(dims))) + [0,1]
                weights = tf.transpose(weights, dims)
                input_tensor = tf.transpose(input_tensor, dims)
                print(weights.get_shape().as_list())
                print(input_tensor.get_shape().as_list())
                print(">>>>>>>>>>> Do Multiply!")
                input_tensor = tf.matmul(weights, input_tensor, name="posetransformation") # [batch, num_ch, height, weight, vec_dim_output, 1]
                input_tensor = tf.squeeze(input_tensor, axis=-1) # [batch, num_ch, height, width, vec_dim_output]
                print(input_tensor.get_shape().as_list())
                print(">>>>>>>>>>> Undo Transpose")
                dims = list(range(len(input_tensor_shape)))
                dims = [dims[0], dims[-1]] + dims[1:len(dims)-1]
                print(dims)
                input_tensor = tf.transpose(input_tensor, dims)
                # new shape: [output_vec_dimensions, 1, batch, num_channels, height, width]
                ####print(">>>>>>> Transpose result")
                ####input_tensor = tf.transpose(input_tensor, dims)
                ####intput_tensor_shape = input_tensor.get_shape().as_list()
                #input_tensor_shape[2] = output_vec_dimension
            # [] [unfinished] - havent add a bias term to Wx, - it should be Wx+b!
            #  NB/ in the original Sabour et al code, the '+b' they refer to in 'Wx_plus_b' is where the bias term, b, corresponds to the bias terms that are passed to the update_routing method
            if apply_bias == True:
                raise NotImplemented
    #return input_tensor, input_tensor_shape
    print(input_tensor.get_shape().as_list())
    return input_tensor


def init_conv_2d(input_images, num_f_maps, scope_name, kernel_size=5):
    with tf.name_scope(scope_name):
        #input images = [batch, height, width]
        input_images = tf.expand_dims(input_images, 3)
        #input images = [batch, height, widht,1]
        kernel_shape = [kernel_size, kernel_size, 1, num_f_maps]
        print(">>>>>>> Create kernel weights")
        with tf.variable_scope(scope_name):
            kernel = variables.weight_variable(kernel_shape)
        print(">>>>>>> Convolve")
        print(input_images.get_shape().as_list())
        output = tf.nn.conv2d(input_images, kernel, [1,1,1,1], "SAME", name="Init2DConv")
        print(output.get_shape().as_list())
        # [batch, height, width, num_f_maps]
        output = tf.transpose(output, [0,3,1,2])
        # [batch, num_f_maps, height, width]
        # Now, expand_dims .t. each feature map is an atom i.e.
        # a 256Dim vector
        output = tf.expand_dims(output, 1)
        # now: [batch, vec_dim=1, num_chanels=num_f_maps, height, width]
        output_tensor_shape = output.get_shape().as_list()

        #[unfinished]
        print(">>>>>>> Create some bias terms for Conv")
        with tf.variable_scope(scope_name):
            bias_shape = output_tensor_shape[:]
            bias_shape[0] = 1
            bias_shape[-1] = 1
            bias_shape[-2] = 1
            print(">>>>>>> Bias creation")
            print(output.get_shape().as_list()) # will output: 200, 256, 1, 28, 28 ater exe
            biases = variables.bias_variable(bias_shape)
            retiled_bias_shape = output_tensor_shape[:] # [batch, vec, num_ch, h, w]
            print(">>>>>>> Create retiling tensor (rank 1)")
            ###shape_batch = tf.reshape(tf.slice(retiled_bias_shape, [0], [1]), [-1])
            ###shape_h_w = tf.reshape(tf.slice(retiled_bias_shape, [3], [2]), [-1])
            ###retiled_bias_shape = tf.concat([shape_batch, [1],[1], shape_h_w], axis=0)
            retiled_bias_shape[1] = 1
            retiled_bias_shape[2] = 1 # [batch, 1, 1, h, w]
            biases = tf.tile(biases, retiled_bias_shape)
            print(">>>>>>> Add bias and conv output")
            output = tf.add(output, biases)
            output = tf.nn.relu(output)
            print(output.get_shape().as_list())
    return output



def convolutional_capsule_layer_OLD(input_tensor, kernel_height, kernel_width, scope_name,output_kernel_vec_dim=8, strides=[1, 1],
convolve_across_channels=False, num_output_channels=None,
kernel_is_vector=False):
    with tf.name_scope(scope_name):
        # input_kernel = [kernel_height, kernel_width, input_tensor:vec_dim]
        # no. feat. maps of convolution = output_kernel_vec_dim
        # Hence: kernel for convolution = [kernel_height, kernel_width, input_tensor:vec_dim output_kernel_vec_dim]
        # input tensor=[batch, vec, num_dim, heigh, width]

        # strides such all be odd numbers
        print(">>>>>>> Setup")
        if(np.mod(kernel_height, 2)==0 | np.mod(kernel_width, 2)==0):
            raise ValueError
        strides_u = strides
        strides = [1,1] + strides[:] + [1]

        input_tensor_shape = input_tensor.get_shape().as_list()
        vec_dim = input_tensor_shape[1]
        #kernel_shape = [kernel_height, kernel_width, input_tensor_shape[1],
        #    output_kernel_vec_dim];


        #input tensor shape = [batch, vec, num_chan, height, width]
        #reshape to = [batch, vec_dim, height, width, num_channel]
        input_tensor = tf.transpose(input_tensor, [0,1,3,4,2])

        if(convolve_across_channels==True):
            _no_input_channels_for_conv = input_tensor_shape[2]
            output_ch = num_output_channels
            if(output_ch==None):
                output_ch = _no_input_channels_for_conv
        else:
            _no_input_channels_for_conv = 1
            output_ch = input_tensor_shape[2]
        #initialise output tesnor of sape = [batch, output_vec, output_ch, height, width]
        print(">>>>>>> Output Initialisation")
        output_tensor_shape = input_tensor_shape[:]
        output_tensor_shape[1] = output_kernel_vec_dim;
        output_tensor_shape[2] = output_ch;
        '''
        output_tensor = tf.zeros(shape=output_tensor_shape, name="ConvCapInit")
        '''
        output_tensors = []

        print(">>>>>>> Loop over channels")
        for o_chan_num in range(output_ch):
            print(">>>>>>>>> Channel %d" % o_chan_num)
            if(convolve_across_channels==True):
                channel_tensor = input_tensor
            else:
                channel_tensor = input_tensor[:,:,:,:,o_chan_num]
            print(channel_tensor.get_shape().as_list())
            with tf.variable_scope(scope_name):
                if(kernel_is_vector==True):
                    kernel_shape = [vec_dim, kernel_height, kernel_width,
                        _no_input_channels_for_conv, output_kernel_vec_dim]
                    kernel = variables.weight_variable(kernel_shape)
                else:
                    kernel_shape = [1, kernel_height, kernel_width,
                        _no_input_channels_for_conv, output_kernel_vec_dim]
                    kernel = variables.weight_variable(kernel_shape)
                    tmp_kernel_tiling = [1]*len(kernel_shape)
                    tmp_kernel_tiling[0] = input_tensor_shape[1]
                    kernel = tf.tile(kernel, tmp_kernel_tiling)

            print(">>>>>>>>>>>>> Conv3D Setup")
            padding_size = np.array([kernel_height, kernel_width])
            padding_size = (padding_size-1)/2
            padding = np.zeros([len(input_tensor_shape), 2])
            padding[2] = np.array([padding_size[0], padding_size[0]])
            padding[3] = np.array([padding_size[1], padding_size[1]])
            #print(padding)
            print(channel_tensor.get_shape().as_list())
            channel_tensor = tf.pad(channel_tensor,padding)
            print(padding)
            print(channel_tensor.get_shape().as_list())
            print(">>>>>>>>>>>>> run convolution")
            #output_tensor_tmp = tf.nn.conv3d(channel_tensor,kernel, strides, "VALID")
            ''' # TEMP: '''
            print(channel_tensor.get_shape().as_list())
            print(output_kernel_vec_dim)
            print([vec_dim, kernel_height, kernel_width])
            print([1] + strides_u)
            output_tensor_tmp = tf.layers.conv3d(channel_tensor,output_kernel_vec_dim,[vec_dim, kernel_height, kernel_width], [1] + strides_u, "VALID", "channels_last")
            print(output_tensor_tmp.get_shape().as_list())
            ''' END TEMP'''
            #output_tensor_tmp = tf.layers.conv3d(channel_tensor,kernel, strides, "VALID", name="Conv3D")
            # ^[batch, 1, height, width, output_vec_dimension]
            print(">>>>>>>>>>>>> run convolution: transpose")
            output_tensor_tmp = tf.transpose(output_tensor_tmp, [0,4,1,2,3])
            print(">>>>>>>>>>>>> output for channel to overall output")
            #output_tensor[:,:,o_chan_num,:,:] = output_tensor_tmp[:]
            print(output_tensor_tmp.get_shape().as_list())
            output_tensors.append(output_tensor_tmp)
        output_tensor = tf.concat(output_tensors, axis=2)


    return output_tensor









def convolutional_capsule_layer(input_tensor, kernel_height, kernel_width, scope_name,output_kernel_vec_dim=8, strides=[1, 1],
num_output_channels=None,
kernel_is_vector=False):
    with tf.name_scope(scope_name):
        input_tensor_shape = input_tensor.get_shape().as_list()
        print("input tensor shape")
        print(input_tensor_shape)
        # [batch, vec_dim, num_ch, h, w]
        if(num_output_channels==None):
            num_output_channels=input_tensor_shape[2]

        def produce_tensor_Vl(channel_number):
            stacked_slices = []
            i_slices = range(0, input_tensor_shape[3]-kernel_height, strides[0])
            j_slices = range(0, input_tensor_shape[4]-kernel_width, strides[1])
            print("Start stack creation")
            for i in i_slices:
                for j in j_slices:
                    begin = [0,0,channel_number,i,j]
                    #end=list(np.array(input_tensor_shape[0:3])+1)+[i+kernel_height, j+kernel_width]
                    this_size = input_tensor_shape[0:2]+[1, kernel_height, kernel_width]
                    strided_slice = tf.slice(input_tensor,begin, this_size)
                    #print(strided_slice.get_shape().as_list())
                    stacked_slices.append(strided_slice)
            stacked_slices = tf.stack(stacked_slices, axis=5)
            print("End stack list creation")
            print(stacked_slices.get_shape().as_list())
            # [batch, vec_dim, 1, k_h, k_w, SLICES]
            new_shape = input_tensor_shape[0:2] + [1,kernel_height, kernel_width, len(i_slices), len(j_slices)]
            print(new_shape)
            stacked_slices = tf.reshape(stacked_slices, shape=new_shape)
            print(stacked_slices.get_shape().as_list())
            stacked_slices = tf.transpose(stacked_slices, [0,5,6,2,1,3,4]) # [M, x', y',  1, vec_dim, k_h, k_w]
            print(stacked_slices.get_shape().as_list())
            new_shape = stacked_slices.get_shape().as_list()
            new_shape = new_shape[0:5] + [new_shape[5]*new_shape[6]]
            stacked_slices = tf.reshape(stacked_slices, shape=new_shape) # [M, x', y',  1, vec_dim, k_h*k_w]
            print(stacked_slices.get_shape().as_list())
            return stacked_slices

        def produce_matrix_Ml(stacked_tensor_shape):
            matrix_weights_shape = [num_output_channels, output_kernel_vec_dim, stacked_tensor_shape[4]]
            with tf.variable_scope(scope_name):
                with tf.variable_scope('chan'+str(channel_number)):
                    matrix = variables.weight_variable(matrix_weights_shape)
            matrix = tf.expand_dims(matrix, axis=0)
            matrix = tf.expand_dims(matrix, axis=0)
            matrix = tf.expand_dims(matrix, axis=0)
            matrix = tf.tile(matrix, stacked_tensor_shape[0:3] + [1, 1, 1]) # [M, x', y',  num_out_ch, output_vec_dim, vec_dim]
            return matrix
        def prerouting(stacked_slices, matrix):
            stacked_slices = tf.tile(stacked_slices, [1,1,1,num_output_channels, 1, 1])
            output = tf.matmul(matrix, stacked_slices) # [M, x', y', num_out_ch, output_vec_dim, k_h*k_w]
            output = tf.transpose(output, [0,4,3,1,2,5]) # M, v_d^l+1, |T^l+1|, x', y', k_h*k_w
            return output
        prerouted_output = []
        for channel_number in list(range(input_tensor_shape[2])):
            print(">>>>>>> Produce tensor stack")
            this_channel_tensor = produce_tensor_Vl(channel_number)
            print(">>>>>>> Produce matrix")
            this_matrix_multiplier = produce_matrix_Ml(this_channel_tensor.get_shape().as_list())
            print(">>>>>>> Perform multiplication")
            prerouted_output.append(prerouting(this_channel_tensor, this_matrix_multiplier))
            print(">>>>>>> Next channel")
        print(">>>>>Finished channel extraction")
        prerouted_output = tf.stack(prerouted_output, axis=6)
        prerouted_output_shape = prerouted_output.get_shape().as_list()
        prerouted_output_shape = prerouted_output_shape[0:5] + [prerouted_output_shape[5]*prerouted_output_shape[6]]
        print(prerouted_output_shape)
        prerouted_output = tf.reshape(prerouted_output, shape=prerouted_output_shape)
        # M, v_d^l+1, |T^l+1|, x', y', k_h*k_w*|T^l|
        prerouted_output_shape2 = prerouted_output.get_shape().as_list()
        prerouted_output_shape2 = prerouted_output_shape2[0:3] + [prerouted_output_shape2[3]*prerouted_output_shape2[4], prerouted_output_shape2[5]]
        prerouted_output = tf.reshape(prerouted_output, shape=prerouted_output_shape2)
        # M, v_d^l+1, |T^l+1|, x'*y', k_h*k_w*|T^l|

        # squash biases
        print(">>>>>Create squash terms")
        squash_bias_shape = prerouted_output_shape2[1:3] + [1, 1] # [ v_d^l+1, |T^l+1|, 1, 1]
        print(squash_bias_shape)
        with tf.variable_scope(scope_name):
            with tf.variable_scope('squash'):
                squash_biases = variables.bias_variable(squash_bias_shape)
        print([1, 1, prerouted_output_shape2[3], 1])
        squash_biases = tf.tile(squash_biases, [1, 1, prerouted_output_shape2[3], 1])
        # [v_d^l+1, |T^l+1|, x'*y', 1]



        print(">>>>> Perform routing")
        patch_shape=[1,1,prerouted_output_shape[-1]]
        print(patch_shape)
        routed_output = patch_based_routing(prerouted_output, scope_name+'/routing', squash_biases=squash_biases,  num_routing=3, patch_shape=patch_shape, patch_stride=[1,1,1],deconvolution_factors=None, bias_channel_sharing=False)
        # M, v_d^l+1, |T^l+1|, x'*y', 1
        print(">>>>> Finished Routing")
        routed_output = tf.squeeze(routed_output)
         # M, v_d^l+1, |T^l+1|, x'*y'
        routed_output_shape = routed_output.get_shape().as_list()
        output_tensor = tf.reshape(routed_output, routed_output_shape[0:3] + [prerouted_output_shape[3], prerouted_output_shape[4]])
        # M, v_d^l+1, |T^l+1|, x', y'
    return output_tensor
