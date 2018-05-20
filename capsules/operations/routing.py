import tensorflow as tf
import numpy as np
from capsule_functions import _squash
# NB/ squash biases init as 0.1

import variables
import sys

def patch_based_routing(input_tensor, scope_name, squash_biases=None,  num_routing=3, patch_shape=[1,5,5], patch_stride=[1,1,1],deconvolution_factors=None, bias_channel_sharing=False):
    # Start Config [config] #
    clear_after_read_for_votes = True
    # why is voting_tensors being store in TensorArray?

    # End Config #
    print(">>>>>>>>> Deconvolution Present?")
    if((deconvolution_factors!=None) and (len(deconvolution_factors)==3)): # None is the same as [0,0,0] ideally
        input_tensor_shape = tf.shape(input_tensor)
        new_input_tensor_shape = np.multiply(np.array(input_tensor_shape[2::]), np.array(deconvolution_factors)+1);
        new_input_tensor_shape = new_input_tensor_shape + np.array(deconvolution_factors)

        new_input_tensor_shape = input_tensor_shape[0:3] + new_input_tensor_shape[:]
        new_input_tensor = np.zeros((*new_input_tensor_shape))

        input_tensor = np.transpose(input_tensor, 2,3,4,0,1)
        new_input_tensor = np.transpose(new_input_tensor, 2,3,4,0,1)
        for i in range(input_tensor_shape[2]):
            for j in range(input_tensor_shape[3]):
                for k in range(input_tensor_shape[4]):
                    this_slice_pos = np.multiply(np.array([i,j,k]), np.array(deconvolution_factors))
                    #this_slice_pos = input_tensor_shape[0:2] + list(this_slice_pos)
                    this_slice_pos = tuple([i,j,k])
                    this_slice_data = input_tensor[this_slice_pos]
                    new_input_tensor[this_slice_pos] = this_slice_data
                    #new_input_tensor[*this_slice_pos] = input_tensor[3::]
        input_tensor = np.transpose(input_tensor, 3,4,0,1,2)
        new_input_tensor = np.transpose(new_input_tensor, 3,4,0,1,2)

        del input_tensor
        input_tensor = new_input_tensor
    print(">>>>>>>>> Calculate Patch size")
    if(patch_shape==None):
        patch_shape=[1, input_tensor_shape[-2], input_tensor_shape[-1]]

    print(">>>>>>>>> Calculate output dimensions")
    input_tensor_shape = input_tensor.get_shape().as_list()
    output_heightwise = np.floor((input_tensor_shape[-2] - patch_shape[1] + 1)/patch_stride[1])
    output_widthwise = np.floor((input_tensor_shape[-1] - patch_shape[2] + 1)/patch_stride[2])
    output_channelwise = np.floor((input_tensor_shape[2] - patch_shape[0] + 1)/patch_stride[0])
    output_dimensions = [output_channelwise, output_heightwise, output_widthwise]
    output_dimensions = [int(x) for x in output_dimensions]

    # NEED TO RESHAPE THE INPUT_TENSOR:
    # Go From [batch, vec, num_channels, height, width]  ---->
    # To  [batch, vec, patch_num_channels, patch_height, patch_width, output_num_channels, output_height, output_width]


    print(">>>>>>>>> Formulate Input tensor with patch channels")
    input_patch_based_tensor_shape = input_tensor_shape[0:2] + patch_shape + output_dimensions
    #input_patch_based_tensor = np.zeros(input_patch_based_tensor_shape)
    input_patch_based_tensor = []
    print(">>>>>>>>>>> Enter loop")
    for i in range(int(output_heightwise)):
        for j in range(int(output_widthwise)):
            for k in range(int(output_channelwise)):
                sys.stdout.write("Assignment for Slice %d, %d, %d of %d, %d, %d\r" % (i,j,k,int(output_heightwise),int(output_widthwise),int(output_channelwise)))
                sys.stdout.flush()
                #sys.stdout.write(">>>>>>>>>>>>> Get slice from input tensor for assignment to patch based input tensor")
                ii = i*patch_stride[1]
                jj = j*patch_stride[2]
                kk = k*patch_stride[0]
                #this_patch = input_tensor[:,:,kk:kk+patch_shape[0],ii:ii+patch_shape[1],jj:jj+patch_shape[2]]
                begin=[0,0,kk,ii,jj]
                size=[input_patch_based_tensor_shape[0], input_patch_based_tensor_shape[1]] + patch_shape
                this_patch = tf.slice(input_tensor, begin, size)
                # ^ [batch, vec, patch_shape[0](numchannels), patch_shape[1](width), patch_shape[2](height)]
                #this_patch = np.expand_dims(this_patch, axis=-1)
                #this_patch = np.expand_dims(this_patch, axis=-1)
                this_patch_shape = this_patch.get_shape().as_list()

                this_patch = tf.expand_dims(this_patch, axis=-1) # [batch, vec,patchshape[0], patchshape[1], patchshape[2],
                                                                 # 1]
                #sys.stdout.write(">>>>>>>>>>>>> Assignment")
                #input_patch_based_tensor[:,:,:,:,:,i,j,k] = this_patch
                input_patch_based_tensor.append(this_patch)
    print("")
    print(">>>>>>>>> Concat list")
    input_patch_based_tensor = tf.concat(input_patch_based_tensor, axis=5)
    print(input_patch_based_tensor.get_shape().as_list())
    print(input_patch_based_tensor_shape)
    print(">>>>>>>>> Reshape concat")
    input_patch_based_tensor = tf.reshape(input_patch_based_tensor, input_patch_based_tensor_shape)

    with tf.name_scope(scope_name):
        print(">>>>>>>>> TF Input tensor Initialisation")
        #input_patch_based_tensor = tf.constant(input_patch_based_tensor, shape=input_patch_based_tensor_shape, verify_shape=True, name="InitPatchTensor")
        # [batch, vec, patch_num_channels, patch_height, patch_width, output_num_channels, output_height, output_width]


        print(">>>>>>>>> Calculate Logit shape")
        logit_shape_input = input_patch_based_tensor.get_shape().as_list() # [batch, vec, patch_num_channels, patch_height, patch_width, output_num_channels, output_height, output_width]
        logit_shape_input[1] = 1 # [batch, 1, patch_num_channels, patch_height, patch_width, output_num_channels, output_height, output_width]
        logit_shape = logit_shape_input[:]
        # logit shape:  [batch, 1, patch_num_channels, patch_height, patch_width, output_num_channels, output_height, output_width]

        reduced_logit_shape = logit_shape[:]
        reduced_logit_shape = logit_shape[0:-3] + [np.prod(logit_shape[-3::])]

        vec_dim = input_tensor_shape[1]

        if(squash_biases == None):
            with tf.variable_scope(scope_name):
                bias_dimensions = output_dimensions[:]
                if(bias_channel_sharing==True): # wish to share bias terms across all channels
                    bias_dimensions[0] = 1 # [1, o_h, o_w]
                    squash_biases = variables.bias_variable(shape=[vec_dim]+bias_dimensions, init=0.1) #[vec_dim,1,o_h,o_w]
                    tiling_term = [1] * (tf.size(output_dimensions)+1) # [1,1,1,1]
                    tiling_term[1] = output_dimensions[0] # [1,num_ch,1,1]
                    squash_biases = tf.tile(squash_biases, tiling_term) # [vec_dim, o_num_ch, o_h, o_w]
                else: # create a unique bias term for each output capsule across all channels (and width and height)
                    squash_biases = variables.bias_variable(shape=[vec_dim]+output_dimensions, init=0.1) # [vec_dim, o_num_channels, o_height, o_width]
        print(">>>>>>>>> Dynamic Routing: Start Iterating...")
        def _algorithm(i, logits, voting_tensors):
            print(">>>>>>>>>>>>> Iteration: ")
            print(i)
            print(">>>>>>>>>>>>>>>>> Softmax")
            logits = tf.reshape(logits, shape=reduced_logit_shape) # flatten end dimensions of shape
            c_i = tf.nn.softmax(logits, axis=(len(reduced_logit_shape)-1))
            logits = tf.reshape(logits, shape=logit_shape) # restore shape
            # c_i = [batch, 1, patch_num_channels, patch_height, patch_width, total_num_of_output capsules = product(num_channels_output, height_output, width_output)]
            #input_tensor shape = [batch, vec_dim, num_channels, height, width]
            c_i = tf.reshape(c_i, shape=logit_shape)

            print(">>>>>>>>>>>>>>>>> Tiling")
            print(c_i.get_shape().as_list())
            tmp = [1]*len(logit_shape)
            tmp[1] = vec_dim
            print(reduced_logit_shape)
            print(tmp)
            c_i = tf.tile(c_i, tmp)
            # c_i (retiled) = [batch, vec_dim, patch_num_channels, patch_height, patch_width, num_channels_output, height_output, width_output]

            print(">>>>>>>>>>>>>>>>> Voting")
            batch_n = input_tensor_shape[0]
            print(">>>>>>>>>>>>>>>>>>> Multiply")
            s_j = tf.multiply(c_i, input_patch_based_tensor, name="indv_vote")
            dim_to_sum = list(range(2, len(input_tensor_shape)))
            print(">>>>>>>>>>>>>>>>>>> Reduce Sum")
            s_j = tf.reduce_sum(s_j, axis=dim_to_sum, name="vote_res") # [batch, vec_dim, numcha_o, h_o, w_o]
            print(">>>>>>>>>>>>>>>>>>> Add")
            s_j = tf.add(s_j, tf.transpose(tf.tile(tf.expand_dims(squash_biases, 4), [1,1,1,1, batch_n]), [4,0,1, 2, 3])) # [] [unfinished] - the [1,1,1] needs to be done generically

            print(">>>>>>>>>>>>>>>>> Squashing")
            v_j = _squash(s_j) # provided: [batch, vec_dim, num_ch, h_o, w_o]; same out
            #v_j dimensions = [batch, vec_dim, num_channels_output, height_output, width_output]

            # input patch based tensor:[batch, vec, patch_num_channels, patch_height, patch_width, output_num_channels, output_height, output_width]

            print(">>>>>>>>>>>>>>>>> Vote Tensor Tiling")
            v_j_tiled = tf.expand_dims(v_j, 2)
            v_j_tiled = tf.expand_dims(v_j_tiled, 2)
            v_j_tiled = tf.expand_dims(v_j_tiled, 2)
            # v_j_tiled = [batch, vec, 1, 1, 1 o_numch, o_h_o, o_w_o]
            v_j_tiled = tf.tile(v_j_tiled, [1,1]+patch_shape+[1,1,1])
            # v_j_tiled = [batch, vec, p_numch, p_h_o, p_w_o, o_numch, o_h_o, o_w_o]

            print(">>>>>>>>>>>>>>>>> Cosine similarity")
            dot_product = tf.multiply(input_patch_based_tensor, v_j_tiled, name="dot_prod_multiply")
            # dimensions of dot product: [batch, vec_dim, pnum_channels, pheight, pwidth,,,, onum_channels_output, oheight_output, owidth_output]
            dot_product = tf.reduce_sum(dot_product, axis=[1], keepdims=True,  name="dot_prod_add")
            # dimensions of dot product: [batch,    1   , pnum_channels, pheight, pwidth,,,, onum_channels_output, oheight_output, owidth_output]


            print(">>>>>>>>>>>>>>>>> Logit update")
            # logit shape:  [batch, 1, patch_num_channels, patch_height, patch_width, output_num_channels, output_height, output_width]
            print(logits.get_shape().as_list())
            print(dot_product.get_shape().as_list())
            logits = tf.add(logits, dot_product, name="b_ij_update")
            print(">>>>>>>>>>>>>>>>> Write the output")
            voting_tensors = voting_tensors.write(i, v_j) # original below line wasn't here, and this was uncommented [] [unfinished]
            #voting_tensors = v_j
            print(">>>>>>>>>>>>>>>>> Return")
            return tf.add(i,1), logits, voting_tensors

        logits = tf.fill(logit_shape, 0., name="b_ij_init")
        voting_tensors = tf.TensorArray(size=num_routing, dtype=tf.float32, clear_after_read=clear_after_read_for_votes) # do not clear after read as we want to investigate
        #i = tf.constant(0, dtype=tf.int16) # original below line wasn't here, and this was uncommented [] [unfinished]
        #voting_tensors = tf.zeros(input_tensor_shape[0:2] + output_dimensions, dtype=tf.float32, name="voting_init")
        i=tf.constant(0)
        _, logits, voting_tensors = tf.while_loop(
            lambda i, logits, voting_tensors: i < num_routing,
            _algorithm,
            loop_vars = [i, logits, voting_tensors],
            swap_memory=True
        )
        #for i in range(num_routing):
        #    _, logits, voting_tensors = _algorithm(i, logits, voting_tensors)
        print(">>>>>>>>>Finished Routing")

        resulting_votes= voting_tensors.read(num_routing-1)# original below line wasn't here, and this was uncommented [] [unfinished]
        #resulting_votes= voting_tensors
        print(resulting_votes.get_shape().as_list())
    return resulting_votes




def routing(input_tensor, scope_name, output_dimensions=None, squash_biases=None, num_routing=3, bias_channel_sharing=False):
    # This method will not work for patch-based routing!

    # input_tensor shape = [batch, vec_dim, num_channels, height, width]
    # NB/ It is important that the input_tensor shape has the capsule numbers in the last dimensions
    #     as the first two dimensions will always be interpretted as `batch` and `vec_dim` ( where
    #     `vec_dim` is the capsule vector dimensionality )

    # output_dimensions specify the dimensions of the output in the form:
    # [num_channels, height, width]

    # NB/ in Sabour et al. 2017, the code shows that the bias terms added to line 5 of Procedure 1, "Routing algorithm", is only shared across all channels. A bias term for Capsule (i,j) on chanel 1 is the same as the bias term for Capsule (i,j) on channel 2
    # There will be a switch here that can allow or disallow for this

    # NB/ squash_biases are trainable!

    # Start Config [config] #
    clear_after_read_for_votes = True
    gpu_cpu_swap = True
    # why is voting_tensors being store in TensorArray?

    # End Config #

    print(">>>>>>> Initialise logit shape")
    input_tensor_shape = input_tensor.get_shape().as_list()
    logit_shape_input = input_tensor_shape[:]
    logit_shape_input[1] = 1
    logit_shape = logit_shape_input[:] + output_dimensions
    # logit_shape = [batch, 1, num_channels, height, width,  num_channels_output, height_output, width_output]
    print(logit_shape)
    reduced_logit_shape = logit_shape_input + [np.product(output_dimensions)]

    vec_dim = input_tensor_shape[1]


    print(">>>>>>> Squash")
    with tf.name_scope(scope_name):
        if(squash_biases == None):
            with tf.variable_scope(scope_name):
                bias_dimensions = output_dimensions[:]
                if(bias_channel_sharing==True): # wish to share bias terms across all channels
                    bias_dimensions[0] = 1 # [1, o_h, o_w]
                    squash_biases = variables.bias_variable(shape=bias_dimensions) # [1, o_h, o_w]
                    tiling_term = [1] * len(output_dimensions) # [1, 1, 1]
                    tiling_term[0] = output_dimensions[0] # [o_num_ch, 1, 1]
                    squash_biases = tf.tile(squash_biases, tiling_term) # [o_num_ch, o_h, o_w]
                else: # create a unique bias term for each output capsule across all channels (and width and height)
                    squash_biases = variables.bias_variable(shape=output_dimensions) # [num_channels, height, width]

        print(">>>>>>> start defining routing algorithm")
        squash_biases = tf.expand_dims(squash_biases, 3)
        squash_biases = tf.expand_dims(squash_biases, 3) # [num_ch, h, w, 1, 1]
        def _algorithm(i, logits, voting_tensors):
            print(">>>>>>>>>>>>> Iteration:")
            print(i)
            logits = tf.reshape(logits, shape=reduced_logit_shape)
            print(reduced_logit_shape)
            c_i = tf.nn.softmax(logits, axis=len(reduced_logit_shape)-1)
            logits = tf.reshape(logits, shape=logit_shape)
            # c_i = [batch, 1, num_channels, height, width, total_num_of_output capsules = product(num_channels_output, height_output, width_output)]
            #input_tensor shape = [batch, vec_dim, num_channels, height, width]
            print(">>>>>>>>>>>>>>>>> Tiling 1")
            c_i = tf.reshape(c_i, shape=logit_shape)

            tmp = [1]*len(logit_shape)
            tmp[1] = vec_dim
            c_i = tf.tile(c_i, tmp)
            print(">>>>>>>>>>>>>>>>> Tiling 2")
            print(input_tensor.get_shape().as_list())
            print(input_tensor_shape)
            tmp = [1]*len(input_tensor_shape)
            tmp = tmp[:] + output_dimensions
            tmp_tensor = tf.expand_dims(input_tensor, len(input_tensor_shape))
            for ii in range(len(output_dimensions)-1):
                tmp_tensor=tf.expand_dims(tmp_tensor, len(input_tensor_shape))
            tmp_tensor = tf.tile(tmp_tensor, tmp) # dimensoins: [batch, vec_dim, num_channels, height, width,,,, num_channels_output, height_output, width_output]
            print(tmp_tensor.get_shape().as_list())
            print(">>>>>>>>>>>>>>>>> Voting")
            batch = input_tensor_shape[0]
            s_j = tf.multiply(c_i, tmp_tensor, name="indv_vote")
            dim_to_sum = list(range(2, len(input_tensor_shape)))
            s_j = tf.reduce_sum(s_j, axis=dim_to_sum, name="vote_res")
            s_j = tf.add(s_j, tf.transpose(tf.tile(squash_biases, [1,1,1, batch, vec_dim]), [3,4,0,1,2])) # [] [unfinished] - the [1,1,1] needs to be done generically
            # : [batch, vec_dim, num_ch, h, w]
            print(">>>>>>>>>>>>>>>>> Squashing")

            v_j = _squash(s_j)
            #v_j dimensions = [batch, vec_dim, num_channels_output, height_output, width_output]

            print(">>>>>>>>>>>>>>>>> Vote Tensor Tiling 1")
            tmp = [1]*len(input_tensor_shape)
            tmp = tmp[:] + output_dimensions
            tmp_input_tensor = tf.expand_dims(input_tensor, len(input_tensor_shape))
            print(tmp_input_tensor.get_shape().as_list())
            for ii in range(len(output_dimensions)-1):
                tmp_input_tensor=tf.expand_dims(tmp_input_tensor, len(input_tensor_shape))
            tmp_input_tensor = tf.tile(tmp_input_tensor, tmp) # dimensoins: [batch, vec_dim, num_channels, height, width,,,, num_channels_output, height_output, width_output]
            print(">>>>>>>>>>>>>>>>> Vote Tensor Tiling 2")
            tmp = [1] * len(v_j.get_shape().as_list())
            tmp = tmp[:] + input_tensor_shape[2::]
            tmp_vj_tensor = tf.expand_dims(v_j, len(tmp)-len(input_tensor_shape[2::]))
            for ii in range(len(input_tensor_shape[2::])-1):
                tmp_vj_tensor=tf.expand_dims(tmp_vj_tensor, len(tmp)-len(input_tensor_shape[2::]))
            tmp_vj_tensor = tf.tile(tmp_vj_tensor, tmp) # dimensions: [batch, vec_dim, num_channels_output, height_output, width_output, num_channels, height, width]
            print(">>>>>>>>>>>>>>>>> Vote Tensor Transposing")
            #tmp_vj_tensor = tf.transpose(tmp_vj_tensor, [0,1,5,6,7,2,3,4])
            '''tmp_transpose_list = list(range(len(tmp_vj_tensor.get_shape().as_list())))
            tmp_last_elements = tmp_transpose_list[len(v_j.get_shape().as_list())::]
            tmp_first_elements = tmp_transpose_list[2:len(tmp_vj_tensor.get_shape().as_list())]
            tmp_vj_transposer = tmp_last_elements[:] + tmp_first_elements[:]
            tmp_vj_tensor = tf.transpose(tmp_vj_tensor, tmp_vj_transposer) # dimensoins: [batch, vec_dim, num_channels, height, width,,,, num_channels_output, height_output, width_output]'''
            dims = list(range(len(tmp_vj_tensor.get_shape().as_list())))
            dims = dims[0:2] + dims[-3::] + dims[-6:-3]
            tmp_vj_tensor = tf.transpose(tmp_vj_tensor, dims)# dimensoins: [batch, vec_dim, num_channels, height, width,,,, num_channels_output, height_output, width_output]
            print(">>>>>>>>>>>>>>>>> Cosine similarity")
            print(tmp_input_tensor.get_shape().as_list())
            print(tmp_vj_tensor.get_shape().as_list())
            dot_product = tf.multiply(tmp_input_tensor, tmp_vj_tensor, name="dot_prod_multiply")
            # dimensions of dot product: [batch, vec_dim, num_channels, height, width,,,, num_channels_output, height_output, width_output]
            dot_product = tf.reduce_sum(dot_product, axis=[1], keepdims=True,  name="dot_prod_add")
            # dimensions of dot product: [batch,    1   , num_channels, height, width,,,, num_channels_output, height_output, width_output]

            print(">>>>>>>>>>>>>>>>> Logit update")

            # logit_shape = [batch, 1, num_channels, height, width,  num_channels_output, height_output, width_output]
            logits = tf.add(logits, dot_product, name="b_ij_update")

            # [] [unfinished] - algorithm only works when the output_dimensions has the same parameters as the input_tensors, i.e:
            # no change in the num_channels, width or height. this is because of the 'dot product' step present in the algorithm
            print(">>>>>>>>>>>>>>>>> Write the output")

            voting_tensors = voting_tensors.write(i, v_j)
            print(">>>>>>>>>>>>>>>>> Return")

            return (tf.add(i,1), logits, voting_tensors)

        logits = tf.fill(logit_shape, 0., name="b_ij_init")
        voting_tensors = tf.TensorArray(size=num_routing, dtype=tf.float32, clear_after_read=clear_after_read_for_votes) # do not clear after read as we want to investigate
        i = tf.constant(0)
        _, logits, voting_tensors = tf.while_loop(
            lambda i, logits, voting_tensors: i < num_routing,
            _algorithm,
            loop_vars = [i, logits, voting_tensors],
            swap_memory=gpu_cpu_swap
        )
        #for i in range(num_routing):
        #    _, logits, voting_tensors = _algorithm(i, logits, voting_tensors)
        results = voting_tensors.read(num_routing-1)

    return results



def patch_based_routing_OLD(input_tensor, scope_name, squash_biases=None,  num_routing=3, patch_shape=[1,5,5], patch_stride=[1,1,1],deconvolution_factors=None):
    # patch_shape should take dimensions [num_channels, patch_width, patch_height]
    # by default, we do not want cross-channel patching

    # This method will not work for patch-based routing!

    # input_tensor shape = [batch, vec_dim, num_channels, height, width]
    # NB/ It is important that the input_tensor shape has the capsule numbers in the last dimensions
    #     as the first two dimensions will always be interpretted as `batch` and `vec_dim` ( where
    #     `vec_dim` is the capsule vector dimensionality )

    # output_dimensions specify the dimensions of the output in the form:
    # [num_channels, height, width]

    # NB/ in Sabour et al. 2017, the code shows that the bias terms added to line 5 of Procedure 1, "Routing algorithm", is only shared across all channels. A bias term for Capsule (i,j) on chanel 1 is the same as the bias term for Capsule (i,j) on channel 2
    # There will be a switch here that can allow or disallow for this

    # NB/ squash_biases are trainable!

    # Start Config [config] #
    clear_after_read_for_votes = True
    gpu_cpu_swap = True
    # why is voting_tensors being store in TensorArray?

    # End Config #
    print(">>>>>>>>> Deconvolution Present?")
    if((deconvolution_factors!=None) and (len(deconvolution_factors)==3)): # None is the same as [0,0,0] ideally
        input_tensor_shape = tf.shape(input_tensor)
        new_input_tensor_shape = np.multiply(np.array(input_tensor_shape[2::]), np.array(deconvolution_factors)+1);
        new_input_tensor_shape = new_input_tensor_shape + np.array(deconvolution_factors)

        new_input_tensor_shape = input_tensor_shape[0:3] + new_input_tensor_shape[:]
        new_input_tensor = np.zeros((*new_input_tensor_shape))

        input_tensor = np.transpose(input_tensor, 2,3,4,0,1)
        new_input_tensor = np.transpose(new_input_tensor, 2,3,4,0,1)
        for i in range(input_tensor_shape[2]):
            for j in range(input_tensor_shape[3]):
                for k in range(input_tensor_shape[4]):
                    this_slice_pos = np.multiply(np.array([i,j,k]), np.array(deconvolution_factors))
                    #this_slice_pos = input_tensor_shape[0:2] + list(this_slice_pos)
                    this_slice_pos = tuple([i,j,k])
                    this_slice_data = input_tensor[this_slice_pos]
                    new_input_tensor[this_slice_pos] = this_slice_data
                    #new_input_tensor[*this_slice_pos] = input_tensor[3::]
        input_tensor = np.transpose(input_tensor, 3,4,0,1,2)
        new_input_tensor = np.transpose(new_input_tensor, 3,4,0,1,2)

        del input_tensor
        input_tensor = new_input_tensor
    print(">>>>>>>>> Calculate Patch size")
    if(patch_shape==None):
        patch_shape=[1, input_tensor_shape[-2], input_tensor_shape[-1]]

    print(">>>>>>>>> Calculate output dimensions")
    input_tensor_shape = input_tensor.get_shape().as_list()
    output_heightwise = np.floor((input_tensor_shape[-2] - patch_shape[1] + 1)/patch_stride[1])
    output_widthwise = np.floor((input_tensor_shape[-1] - patch_shape[2] + 1)/patch_stride[2])
    output_channelwise = np.floor((input_tensor_shape[2] - patch_shape[0] + 1)/patch_stride[0])
    output_dimensions = [output_channelwise, output_heightwise, output_widthwise]
    output_dimensions = [int(x) for x in output_dimensions]

    # NEED TO RESHAPE THE INPUT_TENSOR:
    # Go From [batch, vec, num_channels, height, width]  ---->
    # To  [batch, vec, patch_num_channels, patch_height, patch_width, output_num_channels, output_height, output_width]


    print(">>>>>>>>> Formulate Input tensor with patch channels")
    input_patch_based_tensor_shape = input_tensor_shape[0:3] + patch_shape + output_dimensions
    input_patch_based_tensor = np.zeros(input_patch_based_tensor_shape)
    for i in range(output_heightwise):
        for j in range(output_widthwise):
            for k in range(output_channelwise):
                ii = i*patch_stride[1]
                jj = j*patch_stride[2]
                kk = k*patch_stride[0]
                sys.stdout.write(">>>>>>>>>>>>> Get slice from input tensor for assignment to patch based input tensor")
                this_patch = input_tensor[:,:,kk:kk+patch_shape[0],ii:ii+patch_shape[1],jj:jj+patch_shape[2]]
                # ^ [batch, vec, patch_shape[0](numchannels), patch_shape[1](width), patch_shape[2](height)]
                this_patch = np.expand_dims(this_patch, axis=-1)
                this_patch = np.expand_dims(this_patch, axis=-1)
                this_patch = np.expand_dims(this_patch, axis=-1) # [batch, vec,patchshape[0], patchshape[1], patchshape[2],
                                                                 # 1,1,1]
                sys.stdout.write(">>>>>>>>>>>>> Assignment")
                input_patch_based_tensor[:,:,:,:,:,i,j,k] = this_patch

    with tf.name_scope(scope_name):
        print(">>>>>>>>> TF Input tensor Initialisation")
        input_patch_based_tensor = tf.constant(input_patch_based_tensor, shape=input_patch_based_tensor_shape, verify_shape=True, name="InitPatchTensor")
        # [batch, vec, patch_num_channels, patch_height, patch_width, output_num_channels, output_height, output_width]


        print(">>>>>>>>> Calculate Logit shape")
        logit_shape_input = input_patch_based_tensor.get_shape().as_list() # [batch, vec, patch_num_channels, patch_height, patch_width, output_num_channels, output_height, output_width]
        logit_shape_input[1] = 1 # [batch, 1, patch_num_channels, patch_height, patch_width, output_num_channels, output_height, output_width]
        logit_shape = logit_shape_input[:]
        # logit shape:  [batch, 1, patch_num_channels, patch_height, patch_width, output_num_channels, output_height, output_width]

        reduced_logit_shape = logit_shape[:]
        reduced_logit_shape = logit_shape[0:-3] + [np.prod(logit_shape[-3::])]

        vec_dim = input_tensor_shape[1]

        if(squash_biases == None):
            with tf.variable_scope(scope_name):
                bias_dimensions = output_dimensions[:]
                if(bias_channel_sharing==True): # wish to share bias terms across all channels
                    bias_dimensions[0] = 1
                    squash_biases = variables.new_bias_variable(shape=bias_dimensions, init=0.1)
                    tiling_term = [1] * tf.size(output_dimensions)
                    tiling_term[0] = output_dimensions[0]
                    squash_biases = tf.tile(squash_biases, tiling_term)
                else: # create a unique bias term for each output capsule across all channels (and width and height)
                    squash_biases = variables.new_bias_variable(shape=[vec_dim]+output_dimensions, init=0.1) # [vec_dim, num_channels, height, width]

        def _algorithm(i, logits, voting_tensors):
            logits = tf.reshape(logits, shape=reduced_logit_shape)
            c_i = softmax(logits, dim=length(reduced_logit_shape-1))
            # c_i = [batch, 1, patch_num_channels, patch_height, patch_width, total_num_of_output capsules = product(num_channels_output, height_output, width_output)]
            #input_tensor shape = [batch, vec_dim, num_channels, height, width]
            c_i = tf.reshape(c_i, shape=logit_shape)

            tmp = [1]*len(reduced_logit_shape)
            tmp[1] = vec_dim
            c_i = tf.tile(c_i, tmp)
            # c_i (retiled) = [batch, vec_dim, patch_num_channels, patch_height, patch_width, num_channels_output, height_output, width_output]

            s_j = tf.multiply(c_i, input_patch_based_tensor, name="indv_vote")
            dim_to_sum = range(2, len(input_tensor_shape))
            s_j = tf.reduce_sum(s_j, axis=dim_to_sum, name="vote_res") # [batch, vec_dim, numcha_o, h_o, w_o]
            s_j = tf.add(s_j, tf.transpose(tf.tile(squash_biases, [1,1,1,1, batch]), [4,0,1, 2, 3])) # [] [unfinished] - the [1,1,1] needs to be done generically
            v_j = _squash(s_j) # provided: [batch, vec_dim, num_ch, h_o, w_o]; same out
            #v_j dimensions = [batch, vec_dim, num_channels_output, height_output, width_output]

            # input patch based tensor:[batch, vec, patch_num_channels, patch_height, patch_width, output_num_channels, output_height, output_width]

            v_j_tiled = tf.expand_dims(v_j, 2)
            v_j_tiled = tf.expand_dims(v_j, 2)
            v_j_tiled = tf.expand_dims(v_j, 2)
            # v_j_tiled = [batch, vec, 1, 1, 1 o_numch, o_h_o, o_w_o]
            v_j_tiled = tf.tile(v_j_tiled, [1,1]+patch_shape+[1,1,1])
            # v_j_tiled = [batch, vec, p_numch, p_h_o, p_w_o, o_numch, o_h_o, o_w_o]

            dot_product = tf.multiply(input_patch_based_tensor, v_j_tiled, name="dot_prod_multiply")
            # dimensions of dot product: [batch, vec_dim, pnum_channels, pheight, pwidth,,,, onum_channels_output, oheight_output, owidth_output]
            dot_product = tf.reduced_sum(dot_product, axis=[1], keepdims=True,  name="dot_prod_add")
            # dimensions of dot product: [batch,    1   , pnum_channels, pheight, pwidth,,,, onum_channels_output, oheight_output, owidth_output]


            # logit shape:  [batch, 1, patch_num_channels, patch_height, patch_width, output_num_channels, output_height, output_width]
            logits = tf.add(logits, dot_product, name="b_ij_update")

            voting_tensors.write(i, v_j)

            return (i+1, logits, voting_tensors)

        logits = tf.fill(logit_shape, 0., name="b_ij_init")
        voting_tensors = tf.TensorArray(size=num_routing, dtype=tf.float32, clear_after_read=clear_after_read_for_votes) # do not clear after read as we want to investigate
        i = tf.constant(0, dtype=tf.int16)
        _, logits, voting_tensors = tf.while_loop(
            lambda i, logits, voting_tensors: i < num_routing,
            _algorithm,
            loop_vars = [i, logits, voting_tensors],
            swap_memory=gpu_cpu_swap
        )
    return voting_tensors.read(num_routing-1)
