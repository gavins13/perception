import tensorflow as tf
import numpy as np
from capsule_functions import _squash
# NB/ squash biases init as 0.1

import variables
import sys

def route_votes(input_tensor, squash_biases,  num_routing=3 , squash_relu=False, squash_relu_leaky=False, softmax_opposite=False, use_squash_bias=True):
    # Start Config [config] #
    its = input_tensor.get_shape().as_list()

    # input tensor has form: [batch, o_vec, o_num_channels, o_h*o_w, TOBEROUTED]  [M, v_d^l+1, |T^l+1|, x'*y', VOTES]
    if(use_squash_bias==True):
        print("begining on patch based conv caps routing")
        print(squash_biases.get_shape().as_list())
        #squash biases has form: # [o_vec_dim, o_num_channel, o_h*o_w, 1]
        print(squash_biases.get_shape().as_list()) # [v_d^l+1, |T^l+1|, x'*y', 1])
        squash_biases = tf.transpose(squash_biases, [3,0,1,2]) # [1, v_d^l+1, |T^l+1|, x'*y'])
        squash_biases = tf.tile(squash_biases, [its[0], 1, 1, 1])

    input_tensor_shape = input_tensor.get_shape().as_list()
    print(input_tensor_shape)

    print(">>>>>>>>> Calculate Logit shape")
    logit_shape_input = input_tensor.get_shape().as_list()
    logit_shape_input[1] = 1
    logit_shape = logit_shape_input[:] # #[batch, 1, o_num_channel, o_h*o_w, TOBEROUTED]
    ls = logit_shape
    print(logit_shape)
    reduced_logit_shape = logit_shape[0:2] + [logit_shape[2]*logit_shape[3], logit_shape[4]] #[batch, 1, o_num_channel*o_h*o_w, TOBEROUTED]
    print(reduced_logit_shape)

    vec_dim = input_tensor_shape[1]
    batch_n = input_tensor_shape[0]

    def _squash_relu(input_tensor, leaky=False): # provided: [batch, vec_dim, num_ch, h_o, w_o]; same out
        '''norm = tf.norm(input_tensor, axis=1, keepdims=True)
        norm_squared = tf.multiply(norm ,norm)
        part_b = tf.divide( input_tensor, norm)
        denom = tf.add(1., norm_squared)
        part_a = tf.divide(norm_squared , denom)
        res = tf.multiply( part_a, part_b  )
        return tf.nn.relu(part_b)'''
        input_tensor = _squash(input_tensor)
        if(leaky==True):
            return tf.nn.leaky_relu(input_tensor)
        else:
            return tf.nn.relu(input_tensor)

    print(">>>>>>>>> Dynamic Routing: Start Iterating...")
    def _algorithm(i, logits, voting_tensors):
        print(">>>>>>>>>>>>> Iteration: ")
        print(i)
        print(">>>>>>>>>>>>>>>>> Softmax")
        #i=tf.Print(i, [i], ": Iteration")


        logits_prep = tf.transpose(logits, [0,1,4, 2,3])#[batch, 1, TOBEROUTED, o_num_channel, o_h*o_w,]
        logits_prep = tf.reshape(logits_prep, [ls[0], ls[1], ls[4], ls[2]*ls[3], 1]) #[batch, 1, TOBEROUTED, o_num_channel*o_h*o_w, 1]
        if(softmax_opposite==True):
            print("NB/ Not the same softmax as in paper")
            c_i = tf.nn.softmax(logits_prep, axis=2) #[batch, 1, TOBEROUTED, o_num_channel*o_h*o_w, 1]
        else:
            c_i = tf.nn.softmax(logits_prep, axis=3)#[batch, 1, TOBEROUTED, o_num_channel*o_h*o_w, 1]
        c_i = tf.reshape(c_i, [ls[0], ls[1], ls[4], ls[2],ls[3]])#[batch, 1, TOBEROUTED, o_num_channel, o_h*o_w,]
        c_i = tf.transpose(c_i, [0,1,3,4,2])#[batch, 1,  o_num_channel, o_h*o_w,TOBEROUTED]


        '''logits = tf.reshape(logits, shape=reduced_logit_shape) # flatten end dimensions of shape
        c_i = tf.nn.softmax(logits, axis=2)
        #c_i = logits
        logits = tf.reshape(logits, shape=logit_shape) # #[batch, 1, o_num_channel, o_h*o_w, TOBEROUTED]
        c_i = tf.reshape(c_i, shape=logit_shape) # #[batch, 1, o_num_channel, o_h*o_w, TOBEROUTED]'''



        '''######################
        #c_i += tf.expand_dims(squash_biases, axis=4) #[batch, v_d^l+1,  o_num_channel, o_h*o_w,TOBEROUTED]
        #c_i = tf.reduce_sum(c_i, axis=1, keepdims=True)#[batch, 1,  o_num_channel, o_h*o_w,TOBEROUTED]
        v_j = tf.multiply(input_tensor, c_i)
        v_j = tf.reduce_mean(v_j, axis=4)
        c_i = tf.reduce_mean(v_j, axis=[1], keepdims=True)
        c_i = tf.tile(c_i, [1,1,1,1,input_tensor_shape[4]])
        voting_tensors = voting_tensors.write(i, v_j)
        print(">>>>>>>>>>>>>>>>> Return")
        return i+1, c_i, voting_tensors############################'''




        print(">>>>>>>>>>>>>>>>> Voting")
        print(">>>>>>>>>>>>>>>>>>> Multiply")
        s_j = tf.multiply(input_tensor, c_i)  # [batch, o_vec, o_num_channels, o_h*o_w, TOBEROUTED]
        print(s_j.get_shape().as_list())
        print(">>>>>>>>>>>>>>>>>>> Reduce Sum")
        s_j = tf.reduce_sum(s_j, axis=4) # [batch, o_vec, o_num_channels, o_h*o_w]
        print(s_j.get_shape().as_list())
        print(">>>>>>>>>>>>>>>>>>> Add")
        #s_j = tf.add(s_j, tf.transpose(tf.tile(tf.expand_dims(squash_biases, 3), [1,1,1, batch_n]), [3,0,1, 2])) # [] [unfinished] - the [1,1,1] needs to be done generically
        # squash bisaes: [1,o_vec_dim, o_num_channel, o_h*o_w]
        s_j = tf.add(s_j, squash_biases) if (use_squash_bias==True) else s_j

        print(s_j.get_shape().as_list())

        print(">>>>>>>>>>>>>>>>> Squashing")
        if(squash_relu==False):
            v_j = _squash(s_j) # provided: [batch, vec_dim, num_ch, h_o, w_o]; same out
        else:
            v_j = _squash_relu(s_j, leaky=squash_relu_leaky)
        #v_j dimensions = [batch, vec_dim, num_channels_output, height_output, width_output]
        # # [batch, o_vec, o_num_channels, o_h*o_w]
        print(v_j.get_shape().as_list())
        v_j = tf.expand_dims(v_j, 4)# [batch, o_vec, o_num_channels, o_h*o_w, 1]
        print(v_j.get_shape().as_list())
        # input patch based tensor:  # [batch, o_vec, o_num_channels, o_h*o_w, TOBEROUTED]

        print(">>>>>>>>>>>>>>>>> Cosine similarity")
        dot_product = tf.multiply(input_tensor, v_j)
        #dot_product  = tf.tile(v_j, [1,1,1,1,input_tensor_shape[4]])
        print(dot_product.get_shape().as_list())
        #  # [batch, o_vec, o_num_channels, o_h*o_w, TOBEROUTED]
        dot_product = tf.reduce_sum(dot_product, axis=1, keepdims=True)
        # dimensions of dot product:  [batch,1, o_num_channels, o_h*o_w, TOBEROUTED]
        print(dot_product.get_shape().as_list())

        print(">>>>>>>>>>>>>>>>> Logit update")
        print(logits.get_shape().as_list())
        print(dot_product.get_shape().as_list())
        logits = tf.add(logits, dot_product)
        print(">>>>>>>>>>>>>>>>> Write the output")
        voting_tensors = voting_tensors.write(i, v_j) # original below line wasn't here, and this was uncommented [] [unfinished]
        #voting_tensors = v_j
        print(">>>>>>>>>>>>>>>>> Return")
        return tf.add(i,1), logits, voting_tensors

    output_shape = input_tensor_shape[:]
    output_shape[-1] = 1

    logits = tf.fill(logit_shape, 0.)
    voting_tensors = tf.TensorArray(size=num_routing, dtype=tf.float32, clear_after_read=True,
    element_shape=output_shape) # do not clear after read as we want to investigate
    #voting_tensors = tf.fill(input_tensor_shape[0:4]+[1], 0., name="v_j_init")

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
    # # [batch, o_vec, o_num_channels, o_h*o_w, 1]
    print(resulting_votes.get_shape().as_list())

    #return tf.cast(resulting_votes, dtype="float32")
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
            swap_memory=True, parallel_iterations=10
        )
        #for i in range(num_routing):
        #    _, logits, voting_tensors = _algorithm(i, logits, voting_tensors)
        results = voting_tensors.read(num_routing-1)

    return results
