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


def init_conv_2d(input_images, num_f_maps, scope_name, kernel_size=5, type="SAME"):
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
        output = tf.nn.conv2d(input_images, kernel, [1,1,1,1], type, name="Init2DConv")
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
kernel_is_vector=False, upsampling_factor=None, type="SAME", num_routing=3):
    with tf.name_scope(scope_name):
        if(type=="SAME"):
            padding_height = int((kernel_height-1)/2)
            padding_width = int((kernel_width-1)/2)
            paddings = [ [0,0], [0,0], [0,0], [padding_height, padding_width], [padding_height, padding_width]]
            input_tensor = tf.pad(input_tensor, paddings )

        input_tensor_shape = input_tensor.get_shape().as_list()
        print("input tensor shape")
        print(input_tensor_shape)
        # [batch, vec_dim, num_ch, h, w]
        if(num_output_channels==None):
            num_output_channels=input_tensor_shape[2]

        def produce_tensor_Vl(channel_number):
            i_slices = range(0, input_tensor_shape[3]-kernel_height+1, strides[0])
            j_slices = range(0, input_tensor_shape[4]-kernel_width+1, strides[1])
            print(list(i_slices))
            print(list(j_slices))
            print(input_tensor_shape)
            i_slices_len = len(list(i_slices))
            j_slices_len = len(list(j_slices))
            print("Start stack creation")



            '''stacked_slices=[]
            for i in i_slices:
                for j in j_slices:
                    begin = [0,0,channel_number,i,j]
                    #end=list(np.array(input_tensor_shape[0:3])+1)+[i+kernel_height, j+kernel_width]
                    this_size = input_tensor_shape[0:2]+[1, kernel_height, kernel_width]
                    strided_slice = tf.slice(input_tensor,begin, this_size)
                    #print(strided_slice.get_shape().as_list())
                    stacked_slices.append(strided_slice)
            stacked_slices = tf.stack(stacked_slices, axis=5)'''
            def _stack(ii,jj,stacked_slices):
                i_slicess = tf.constant(list(i_slices))
                j_slicess = tf.constant(list(j_slices))
                i = i_slicess[ii]
                j = j_slicess[jj]
                #ii = tf.Print(ii, [ii,jj,i,j], 'This done')
                begin = [0,0,channel_number,i,j]
                #end=list(np.array(input_tensor_shape[0:3])+1)+[i+kernel_height, j+kernel_width]
                this_size = input_tensor_shape[0:2]+[1, kernel_height, kernel_width]
                strided_slice = tf.slice(input_tensor,begin, this_size)
                #print(strided_slice.get_shape().as_list())
                #stacked_slices.append(strided_slice)
                ind = tf.add(tf.multiply(ii,j_slices_len), jj)
                #ind = tf.Print(ind, [ind,ii,jj,i,j], 'This done')
                stacked_slices=stacked_slices.write(ind,strided_slice)
                def incrementii():
                    tmp = ii +1
                    tmp2 = tf.constant(0)
                    return tmp, tmp2
                def incrementjj():
                    tmp = tf.add(jj, 1)
                    tmp2 = ii
                    return tmp2, tmp
                ii,jj = tf.cond(tf.equal(jj, (j_slices_len-1)), incrementii, incrementjj)
                #if(tf.equal(jj, (j_slices_len-1))):
                #    ii=ii+1
                #    jj=tf.constant(0)
                #else:
                #    jj=jj+1
                return ii,jj,stacked_slices

            stacked_slices=tf.TensorArray(dtype=tf.float32, size=i_slices_len*j_slices_len, clear_after_read=True)
            current_i=tf.constant(0)
            current_j=tf.constant(0)
            _,_,stacked_slices = tf.while_loop(lambda i,j,stacked_slices: i<i_slices_len,
            _stack, loop_vars = [current_i,current_j,stacked_slices], swap_memory=True, parallel_iterations=1)
            stacked_slices = stacked_slices.stack() # WILL STACK on axis=0, but we need to stack along axis = 5
            #stacked_slices = tf.Print(stacked_slices, [i_slices_len], 'Stacked success')
            stacked_slices = tf.transpose(stacked_slices, [1,2,3,4,5,0])
            stacked_slices_new_shape = input_tensor_shape[0:2] + [1] + [kernel_height, kernel_width, i_slices_len*j_slices_len]
            stacked_slices.set_shape(stacked_slices_new_shape)







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
            if(upsampling_factor!=None):
                stacked_slices = tf.tile(stacked_slices, [1, upsampling_factor, upsampling_factor, 1, 1, 1])
            return stacked_slices



        with tf.variable_scope(scope_name):
            with tf.variable_scope('channel_weights'):
                matrix_weights_shape = [num_output_channels, output_kernel_vec_dim, input_tensor_shape[1], input_tensor_shape[2]] #  num_out_ch, output_vec_dim, vec_dim, INPUT_CHANNELS]
                matrix = variables.weight_variable(matrix_weights_shape)

        def produce_matrix_Ml(stacked_tensor_shape, channel_number):
            matrix_weights_shape_for_channel = [num_output_channels, output_kernel_vec_dim, stacked_tensor_shape[4]]
            print(">>>>>>>>>>>>>>>> Matrix for this Channel Tensor")
            print(channel_number)
            #with tf.variable_scope(scope_name):
            #    with tf.variable_scope('channel_weights', reuse=True):
            #        matrix = variables.weight_variable(matrix_weights_shape)
            this_matrix = tf.squeeze(tf.slice(matrix, [0,0,0,channel_number], matrix_weights_shape_for_channel+[1]), axis=3)
            this_matrix = tf.expand_dims(this_matrix, axis=0)
            this_matrix = tf.expand_dims(this_matrix, axis=0)
            this_matrix = tf.expand_dims(this_matrix, axis=0)
            this_matrix = tf.tile(this_matrix, stacked_tensor_shape[0:3] + [1, 1, 1]) # [M, x', y',  num_out_ch, output_vec_dim, vec_dim]
            return this_matrix
        def prerouting(stacked_slices, matrix):
            stacked_slices = tf.tile(stacked_slices, [1,1,1,num_output_channels, 1, 1])
            output = tf.matmul(matrix, stacked_slices) # [M, x', y', num_out_ch, output_vec_dim, k_h*k_w]
            output = tf.transpose(output, [0,4,3,1,2,5]) # M, v_d^l+1, |T^l+1|, x', y', k_h*k_w
            return output


        def _channel_generator(channel_number, prerouted_output):
            print(">>>>>>> Produce tensor stack")
            this_channel_tensor = produce_tensor_Vl(channel_number)
            print(this_channel_tensor.get_shape().as_list())
            print(">>>>>>> Produce matrix")
            print(channel_number)
            this_matrix_multiplier = produce_matrix_Ml(this_channel_tensor.get_shape().as_list(), channel_number)
            print(this_matrix_multiplier.get_shape().as_list())
            print(">>>>>>> Perform multiplication")
            #prerouted_output.append(prerouting(this_channel_tensor, this_matrix_multiplier))
            this_output=prerouting(this_channel_tensor, this_matrix_multiplier)
            print(this_output.get_shape().as_list())
            prerouted_output = prerouted_output.write(channel_number, this_output )
            print(">>>>>>> Next channel")
            return channel_number+ 1, prerouted_output
        #prerouted_output = []
        prerouted_output = tf.TensorArray(dtype=tf.float32,size=input_tensor_shape[2], clear_after_read=True) # [] double check clear_after_read
        _, prerouted_output = tf.while_loop(lambda channel_number, prerouted_output: channel_number <input_tensor_shape[2],
            _channel_generator, loop_vars = [tf.constant(0), prerouted_output],
            swap_memory=True #parallel_iterations=1???
        )
        prerouted_output = prerouted_output.stack() # will stack along axis=0 but I need it on axis=6 (the last axis)
        prerouted_output = tf.transpose(prerouted_output, [1,2,3,4,5,6,0])
        tmp = prerouted_output.get_shape().as_list()
        print(tmp)
        prerouted_output.set_shape(tmp[0:6] + [input_tensor_shape[2]])
        print(prerouted_output.get_shape().as_list())
        '''
        prerouted_output = []
        for channel_number in list(range(input_tensor_shape[2])):
            print(">>>>>>> Produce tensor stack")
            this_channel_tensor = produce_tensor_Vl(channel_number)
            print(">>>>>>> Produce matrix")
            this_matrix_multiplier = produce_matrix_Ml(this_channel_tensor.get_shape().as_list(), channel_number)
            print(">>>>>>> Perform multiplication")
            prerouted_output.append(prerouting(this_channel_tensor, this_matrix_multiplier))
            print(">>>>>>> Next channel")
        print(">>>>>Finished channel extraction")
        prerouted_output = tf.stack(prerouted_output, axis=6)'''



        print(">>>>>Finished channel extraction")
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
        #routed_output = patch_based_routing(prerouted_output, scope_name+'/routing', squash_biases=squash_biases,  num_routing=num_routing, patch_shape=patch_shape, patch_stride=[1,1,1],deconvolution_factors=None, bias_channel_sharing=False)
        routed_output = patch_based_routing_for_convcaps(prerouted_output, scope_name+'/routing', squash_biases=squash_biases,  num_routing=num_routing)
        #routed_output = tf.reduce_mean(prerouted_output, axis=4, keepdims=True)
        # M, v_d^l+1, |T^l+1|, x'*y', 1
        print(">>>>> Finished Routing")
        routed_output = tf.squeeze(routed_output, axis=4)
         # M, v_d^l+1, |T^l+1|, x'*y'
        routed_output_shape = routed_output.get_shape().as_list()
        print(routed_output_shape)
        print(routed_output_shape[0:3] + [prerouted_output_shape[3], prerouted_output_shape[4]])
        output_tensor = tf.reshape(routed_output, routed_output_shape[0:3] + [prerouted_output_shape[3], prerouted_output_shape[4]])
        # M, v_d^l+1, |T^l+1|, x', y'
    return output_tensor





def convolutional_capsule_layer_v2(input_tensor, kernel_height, kernel_width, scope_name,output_kernel_vec_dim=8, strides=[1, 1], num_output_channels=None, type="SAME", num_routing=3, split_routing=False, use_matrix_bias=True, use_squash_bias=True):
    print(">>>> %s START" % scope_name)
    with tf.name_scope(scope_name):
        '''if(type=="SAME"):
            padding_height = int((kernel_height-1)/2)
            padding_width = int((kernel_width-1)/2)
            paddings = [ [0,0], [0,0], [0,0], [padding_height, padding_width], [padding_height, padding_width]]
            input_tensor = tf.pad(input_tensor, paddings )'''

        input_tensor_shape = input_tensor.get_shape().as_list()
        its = input_tensor_shape
        print("input tensor shape")
        print(input_tensor_shape)
        # [batch, vec_dim, num_ch, h, w]


        patches = tf.transpose(input_tensor, [0,3,4, 1,2])
        patches = tf.reshape(patches, [its[0], its[3], its[4], its[1]*its[2], 1 ]) # [M, x, y, v_d^l|T^l|, 1 ]
        patches = tf.squeeze(patches, axis=4) #  [M, x, y, v_d^l|T^l|]
        print("Testing")
        print(patches.get_shape().as_list())
        patches = tf.transpose(patches, [1,2,3,0])
        print(patches.get_shape().as_list())
        patches = tf.reshape(patches, [its[3], its[4], its[1]*its[2]*its[0]]) #  [ x, y, v_d^l|T^l|*M]
        print(patches.get_shape().as_list())
        patches = tf.transpose(patches, [2,0,1])#  [ v_d^l|T^l|*M, x, y]
        print(patches.get_shape().as_list())

        patches = tf.expand_dims(patches, axis=3)#  [ v_d^l|T^l|*M, x, y, 1]
        print(patches.get_shape().as_list())

        patches = tf.extract_image_patches(patches, [1,kernel_height, kernel_width, 1], strides=[1]+strides+[1], rates=[1,1,1,1], padding=type) #  [ v_d^l|T^l|*M, x, y, k_w*k_h]
        patches_shape = patches.get_shape().as_list()
        its[3] = patches_shape[1]
        its[4] = patches_shape[2]
        '''if(type=="VALID"):
            slicesize = patches.get_shape().as_list()
            slicesize[1] = slicesize[1] - (kernel_height) + 1
            slicesize[2] = slicesize[2] - (kernel_width) +1
            patches = tf.slice(patches, [0,0,0,0], slicesize)
        patches = tf.tile(patches, [1,1,1,kernel_height*kernel_width]) # [] to delete and uncommment above'''

        print("patches extrateced")
        print(patches.get_shape().as_list())


        patches_shape = patches.get_shape().as_list()
        itsv2 = its[:]
        itsv2[3] = patches_shape[1]
        itsv2[4] = patches_shape[2]

        #  [ v_d^l|T^l|*M, x, y, k_h*k_w]
        patches = tf.expand_dims(patches,axis=4)#  [ v_d^l|T^l|*M, x, y, k_h*k_w,1]
        patches = tf.reshape(patches, [its[1]*its[2]*its[0], itsv2[3], itsv2[4], kernel_height, kernel_width])#  [ v_d^l|T^l|*M, x, y, k_h,k_w]
        patches = tf.transpose(patches, [1,2,3,4,0])#  [ x, y, k_h,k_w, v_d^l|T^l|*M]
        patches = tf.expand_dims(patches, axis=5)
        patches = tf.reshape(patches, [itsv2[3], itsv2[4], kernel_height, kernel_width, its[1]*its[2], its[0]] ) #  [ x, y, k_h,k_w, v_d^l|T^l|, M]
        patches=tf.transpose(patches, [5,0,1,2,3,4])#  [M, x, y, k_h,k_w, v_d^l|T^l|]
        patches = tf.transpose(patches, [0,3,4, 5,1,2]) # [M, k_h,k_w, v_d^l|T^l|, x, y ]
        patches_shape = patches.get_shape().as_list()
        print(patches_shape)


        patches = tf.reshape(patches, [its[0], kernel_height, kernel_width,its[1]*its[2], itsv2[3]*itsv2[4], 1 ]) ## [M, k_h,k_w, v_d^l|T^l|, xy=p, 1 ]
        patches=tf.squeeze(patches, axis=5)# [M, k_h,k_w, v_d^l|T^l|, xy=p ]


        # GOAL: # [M, k_h, k_w, v_d^l|T^l|, p]

        p = int(patches.get_shape().as_list()[4]) # this is supposed to equal its[3]*its[4] or the reduced version = xy=p
        print(p)

        print("this stage")
        print(patches.get_shape().as_list())

        patches = tf.expand_dims(patches, axis=4)
        patches = tf.reshape(patches, [its[0], kernel_height, kernel_width, its[1]*its[2], p]) # [M, k_h, k_w, v_d^l|T^l|, p]
        print(patches.get_shape().as_list())

        patches = tf.transpose(patches, [0,1,2,4, 3]) # [M, k_h, k_w, p, v_d^l|T^l|]
        patches = tf.expand_dims(patches, axis=5) # [M, k_h, k_w, p, v_d^l|T^l|, 1]
        print(patches.get_shape().as_list())
        patches = tf.reshape(patches, [its[0], kernel_height, kernel_width, p, its[1], its[2] ]) # [M, k_h, k_w, p, v_d^l, |T^l|]
        patches = tf.transpose(patches, [0,1,2,3,5,4])  # [M, k_h, k_w, p,  |T^l|, v_d^l]
        patches = tf.expand_dims(patches, 6)# [M, k_h, k_w, p,  |T^l|, v_d^l, 1]
        patches = tf.expand_dims(patches, 5)# [M, k_h, k_w, p,  |T^l|, 1, v_d^l, 1]
        print(patches.get_shape().as_list())
        patches = tf.tile(patches, [1,1,1,1,1,num_output_channels,1,1])# [M, k_h, k_w, p,  |T^l|, |T^l+1|, v_d^l, 1]
        print(patches.get_shape().as_list())
        patches_shape = patches.get_shape().as_list()

        with tf.variable_scope(scope_name):
            matrix_shape = patches_shape[:] # [M, k_h, k_w, p,  |T^l|, |T^l+1|, v_d^l, 1]
            matrix_shape[-1] = patches_shape[-2] # [M, k_h, k_w, p,  |T^l|, |T^l+1|, v_d^l, v_d^l]
            matrix_shape[-2] = output_kernel_vec_dim # [M, k_h, k_w, p,  |T^l|, |T^l+1|, v_d^l+1, v_d^l]

            matrix_shape[0] = 1 # [1, k_h, k_w, p,  |T^l|, |T^l+1|,  v_d^l+1, v_d^l]
            matrix_shape[3] = 1 # [1, k_h, k_w,  1,  |T^l|, |T^l+1|,  v_d^l+1, v_d^l]
            matrix = variables.weight_variable(matrix_shape) # to keep [] to segment
            if(use_matrix_bias==True):
                matrix_shape_bias = matrix_shape[:]
                matrix_shape_bias[6] = 1
                matrix_shape_bias[7] = 1 # [1, k_h, k_w,  1,  |T^l|, |T^l+1|,  1, 1]
                with tf.variable_scope('matrix_bias'):
                    matrix_bias = variables.bias_variable(matrix_shape_bias)
        matrix = tf.tile(matrix, [patches_shape[0], 1, 1,  patches_shape[3]] + [1,1,1,1])
        if(use_matrix_bias==True):
            matrix_bias = tf.tile(matrix_bias, [patches_shape[0], 1, 1,  patches_shape[3]] + [1,1,output_kernel_vec_dim,1])
            result = tf.matmul(matrix, patches) + matrix_bias # [M, k_h, k_w, p,  |T^l|, |T^l+1|,  v_d^l+1, 1]
        else:
            result = tf.matmul(matrix, patches)
        #result = patches # [] todlete

        result = tf.transpose(result, [0,6,5,3,  1,2,4,7])# [M,v_d^l+1, |T^l+1|,p, k_h, k_w,   |T^l|,    1]
        print("result")
        print(result.get_shape().as_list())
        result = tf.reshape(result, [its[0], output_kernel_vec_dim, num_output_channels, p, kernel_height*kernel_width*its[2]*1, 1, 1, 1])
        # [M,v_d^l+1, |T^l+1|,p, k_h* k_w*|T^l|*1, 1, 1    1]
        prerouted_output = tf.squeeze(result, axis=[5,6,7]) # # [M,v_d^l+1, |T^l+1|,p, k_h*k_w*|T^l|]
        prerouted_output_shape = prerouted_output.get_shape().as_list()
        prerouted_output_shape2 = prerouted_output.get_shape().as_list()
        pos = prerouted_output.get_shape().as_list()
        print(prerouted_output_shape)

        # squash biases
        print(">>>>>Create squash terms")
        '''squash_bias_shape = prerouted_output_shape2[1:3] + [1, 1] # [ v_d^l+1, |T^l+1|, 1, 1]'''
        squash_bias_shape = prerouted_output_shape2[1:3] + [prerouted_output_shape2[3], 1] # [ v_d^l+1, |T^l+1|, x'*y', 1]
        print(squash_bias_shape)
        if(use_squash_bias==True):
            with tf.variable_scope(scope_name):
                squash_biases = variables.bias_variable(squash_bias_shape)
        else:
                squash_biases = tf.fill(squash_bias_shape, 0.)
        print([1, 1, prerouted_output_shape2[3], 1])
        '''squash_biases = tf.tile(squash_biases, [1, 1, prerouted_output_shape2[3], 1])'''
        # [v_d^l+1, |T^l+1|, x'*y', 1]
        '''squash_bias_shape = prerouted_output_shape2[1:3] + [ prerouted_output_shape2[3], 1] # [ v_d^l+1, |T^l+1|, x'*y', 1]
        squash_biases = tf.fill(squash_bias_shape, 0.)'''
        # [v_d^l+1, |T^l+1|, x'*y', 1]



        print(">>>>> Perform routing")
        patch_shape=[1,1,prerouted_output_shape[-1]]
        print(patch_shape)
        #routed_output = patch_based_routing(prerouted_output, scope_name+'/routing', squash_biases=squash_biases,  num_routing=num_routing, patch_shape=patch_shape, patch_stride=[1,1,1],deconvolution_factors=None, bias_channel_sharing=False)
        if(split_routing==False):
            routed_output = patch_based_routing_for_convcaps(prerouted_output, squash_biases=squash_biases,  num_routing=num_routing)
        else:
            def _alg(position_number, routed_output):
                squash_slice = tf.slice(squash_biases, [0,0,position_number,0], [squash_bias_shape[0], squash_bias_shape[1], 1, 1])
                prerouted_slice = tf.slice(prerouted_output, [0, 0, 0, position_number, 0], pos[0:3] + [1, pos[4]])
                print(squash_slice.get_shape().as_list())
                print(prerouted_slice.get_shape().as_list())
                out = patch_based_routing_for_convcaps(prerouted_slice, squash_biases=squash_slice, num_routing=num_routing)
                # [M,v_d^l+1, |T^l+1|,1, 1]
                print(">>>>>> Finished _alg")
                routed_output.write(position_number, out)
                return position_number+1, routed_output
            routed_output = tf.TensorArray(size=pos[4], dtype=tf.float32, clear_after_read=True)
            i=tf.constant(0)
            _, routed_output = tf.while_loop(
                lambda i, routed_output: i < pos[4],
                _alg,
                loop_vars = [i, routed_output],
                swap_memory=True
            )
            print(">>>>>> stack")
            routed_output = routed_output.stack() # stacks along axis=0 # [p, M,v_d^l+1, |T^l+1|,1, 1]
            print(">>>>>> set shape")
            routed_output.set_shape([pos[3], pos[0], pos[1], pos[2], 1, 1])
            # we need # [M,v_d^l+1, |T^l+1|,p, 1]
            print(">>>>>>Some shapes")
            print(routed_output.get_shape().as_list())
            routed_output = tf.transpose(routed_output, [1,2,3,0,4,5])
            routed_output = tf.squeeze(routed_output, axis=5)# [M,v_d^l+1, |T^l+1|,p, 1]
            print(routed_output.get_shape().as_list())

        #routed_output = tf.reduce_mean(prerouted_output, axis=4, keepdims=True)
        # M, v_d^l+1, |T^l+1|, x'*y', 1
        print(">>>>> Finished Routing")
        #routed_output = tf.squeeze(routed_output, axis=4)
         # M, v_d^l+1, |T^l+1|, x'*y', 1
        routed_output_shape = routed_output.get_shape().as_list()
        print(routed_output_shape)

        '''if(type=="SAME"):
            _end = [itsv2[3],itsv2[4]]
        else:
            _end = [1, p]
        print(_end)'''
        _end = [itsv2[3],itsv2[4]]

        output_tensor = tf.reshape(routed_output, routed_output_shape[0:3] + _end)
        # M, v_d^l+1, |T^l+1|, x', y'
        print(">>>>> Finished ConvCaps")
        print(output_tensor.get_shape().as_list())
    print(output_tensor.get_shape().as_list())
    print(">>>> %s END" % scope_name)
    return output_tensor



























def simple_capsules(input_tensor, k=5, stride=3, output_shape=[None, 8, 32, 40, 40], pipeline_channels=8, upsampling=None, scope_name='simple', type="SAME", num_routing=3):
    with tf.name_scope(scope_name):
        with tf.variable_scope(scope_name):
            input_tensor = depthwise_convolutional_capsule_layer(input_tensor, k, k, 'conv', strides=[stride, stride], num_output_channels=pipeline_channels, upsampling_factor=upsampling, type=type, conv_vector=False)
            input_tensor = matmul_capsule_layer(input_tensor, 'matmul', output_kernel_vec_dim=output_shape[1])
            input_tensor = quick_routing_2(input_tensor, 'routing', output_shape[2], output_shape[3], output_shape[4], num_routing=num_routing)
    return input_tensor

def depthwise_convolutional_capsule_layer(input_tensor, kernel_height, kernel_width, scope_name, strides=[1, 1], num_output_channels=None, upsampling_factor=None, type="SAME", conv_vector=False, relu=False):
    print(">>>> %s START" % scope_name)
    with tf.name_scope(scope_name):
        ''' Note: if conv_vec=True and type=VALID, then the output vector size is 1 '''
        input_tensor_shape = input_tensor.get_shape().as_list()
        its = input_tensor_shape
        print("input tensor shape depthwise conv")
        print(input_tensor_shape)
        print(input_tensor.dtype)
        # [batch, vec_dim, num_ch, h, w]

        input_tensor = tf.transpose(input_tensor, [0,1,3,4, 2]) # # [batch, vec_dim, h, w, num_ch]
        if(upsampling_factor!=None):
            kernel_shape = [1, kernel_height, kernel_width,  num_output_channels, its[2]]
        else:
            kernel_shape = [1, kernel_height, kernel_width, its[2], num_output_channels]
        print("kernel shape")
        print(kernel_shape)
        if(conv_vector==True):
            kernel_shape[0] = its[1]

        if(upsampling_factor==None):
            input_tensor = tf.layers.conv3d(input_tensor, num_output_channels, kernel_shape[0:3], strides=[1]+strides, padding='same')
        else:
            deconv_shape = input_tensor.get_shape().as_list()
            deconv_shape[2] = int(deconv_shape[2]*upsampling_factor)
            deconv_shape[3] = int(deconv_shape[3]*upsampling_factor)
            deconv_shape[4] = num_output_channels
            print("decov shape")
            print(deconv_shape)
            input_tensor = tf.layers.conv3d_transpose(input_tensor, num_output_channels, kernel_shape[0:3], strides=[1]+strides, padding='same')
        # input tensor now has shape:
        # [batch, vec_dim or 1, h*, w*, o_num_ch]
        input_tensor = tf.transpose(input_tensor, [0,1,4, 2,3])
    print(input_tensor.get_shape().as_list())
    print(">>>> %s END" % scope_name)
    input_tensor = tf.nn.relu(input_tensor) if relu==True else input_tensor
    return input_tensor

def depthwise_convolutional_capsule_layer_NOTKERAS(input_tensor, kernel_height, kernel_width, scope_name, strides=[1, 1], num_output_channels=None, upsampling_factor=None, type="SAME", conv_vector=False, relu=False):
    print(">>>> %s START" % scope_name)
    with tf.name_scope(scope_name):
        ''' Note: if conv_vec=True and type=VALID, then the output vector size is 1 '''
        input_tensor_shape = input_tensor.get_shape().as_list()
        its = input_tensor_shape
        print("input tensor shape depthwise conv")
        print(input_tensor_shape)
        print(input_tensor.dtype)
        # [batch, vec_dim, num_ch, h, w]

        input_tensor = tf.transpose(input_tensor, [0,1,3,4, 2]) # # [batch, vec_dim, h, w, num_ch]
        if(upsampling_factor!=None):
            kernel_shape = [1, kernel_height, kernel_width,  num_output_channels, its[2]]
        else:
            kernel_shape = [1, kernel_height, kernel_width, its[2], num_output_channels]
        print("kernel shape")
        print(kernel_shape)
        if(conv_vector==True):
            kernel_shape[0] = its[1]
        with tf.variable_scope(scope_name):
            kernel = variables.weight_variable(kernel_shape)
        strides = [1,1] + strides + [1]
        if(upsampling_factor==None):
            input_tensor = tf.nn.conv3d(input_tensor, kernel, strides, type)
        else:
            deconv_shape = input_tensor.get_shape().as_list()
            deconv_shape[2] = int(deconv_shape[2]*upsampling_factor)
            deconv_shape[3] = int(deconv_shape[3]*upsampling_factor)
            deconv_shape[4] = num_output_channels
            print("decov shape")
            print(deconv_shape)
            input_tensor = tf.nn.conv3d_transpose(input_tensor, kernel, deconv_shape, strides, type)
        # input tensor now has shape:
        # [batch, vec_dim or 1, h*, w*, o_num_ch]
        input_tensor = tf.transpose(input_tensor, [0,1,4, 2,3])
    print(input_tensor.get_shape().as_list())
    print(">>>> %s END" % scope_name)
    input_tensor = tf.nn.relu(input_tensor) if relu==True else input_tensor
    return input_tensor

def matmul_capsule_layer(input_tensor, scope_name,output_kernel_vec_dim=8, intra_channel_sharing=False):
    with tf.name_scope(scope_name):
        #input tensor:  [batch, vec_dim, num_ch, h, w]
        its = input_tensor.get_shape().as_list()
        print("matmul caps")
        print(its)
        input_tensor = tf.transpose(input_tensor, [0,2,3,4,1])
        input_tensor = tf.expand_dims(input_tensor, axis=5) # [b,num_ch,h,w,v,1]
        matrix_shape = [1, its[2], its[3], its[4], output_kernel_vec_dim,  its[1]]
        matrix_tiling = [its[0], 1, 1, 1, 1, 1]
        if(intra_channel_sharing==True):
            matrix_shape[2] = 1
            matrix_tiling[2] = its[3]
            matrix_shape[3] = 1
            matrix_tiling[3] = its[4]
        with tf.variable_scope(scope_name):
            matrix = variables.weight_variable(matrix_shape)
        matrix = tf.tile(matrix, matrix_tiling)# [b,num_ch,h,w,o_v,v]
        output_tensor = tf.matmul(matrix, input_tensor)# [b,num_ch,h,w,o_v,1]
        output_tensor = tf.squeeze(output_tensor, axis=5)
        output_tensor = tf.transpose(output_tensor, [0,4,1,2,3])
    return output_tensor


def quick_routing_2(input_tensor, scope_name,output_channels, output_height, output_width, num_routing=3):
    ''' Note: this will route all channels as well '''
    with tf.name_scope(scope_name):
        its = input_tensor.get_shape().as_list()
        #input tensor [b,v,ch,h,w]
        print('quick routing')
        print(its)
        output_tensor = routing(input_tensor, 'quickrouting', output_dimensions=[output_channels,output_height,output_width],  num_routing=num_routing) # [b,v,o_ch, p,1]
        #output_tensor = tf.reshape(output_tensor, [its[0], its[1], output_channels, output_height, output_width])
    return output_tensor

def quick_routing_1(input_tensor, scope_name,output_channels, output_height, output_width, num_routing=3):
    ''' Note: this will route all channels as well '''
    with tf.name_scope(scope_name):
        its = input_tensor.get_shape().as_list()
        #input tensor [b,v,ch,h,w]
        print('quick routing')
        print(its)
        input_tensor = tf.expand_dims(input_tensor, axis=2)
        input_tensor = tf.expand_dims(input_tensor, axis=2) # [b,v,1,1,ch,h,w]
        input_tensor = tf.reshape(input_tensor, shape=[its[0], its[1], 1, 1, its[2]*its[3]*its[4]]) # [b,v,1,1,ch*h*w]
        p = output_height*output_width
        input_tensor = tf.tile(input_tensor, [1,1,output_channels, p, 1])#[b,v,o_ch,p,ch*h*w]
        biases_shape = [its[1], output_channels, 1, 1] # [v, o_ch, 1, 1]
        biases_tiling = [ 1, 1, p, 1]
        with tf.variable_scope(scope_name):
            biases = variables.bias_variable(biases_shape)
        biases = tf.tile(biases, biases_tiling) # [v, o_ch, p, 1]
        print('bisaes shape')
        print(biases.get_shape().as_list())
        print('to routing shape')
        print(input_tensor.get_shape().as_list())
        output_tensor = patch_based_routing_for_convcaps(input_tensor, squash_biases=biases,  num_routing=num_routing) # [b,v,o_ch, p,1]
        output_tensor = tf.reshape(output_tensor, [its[0], its[1], output_channels, output_height, output_width])
    return output_tensor
