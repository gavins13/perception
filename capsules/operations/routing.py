import tensorflow as tf

# NB/ squash biases will probably start out as 0.1


# when defining the network:
# will write it out like this:
# layer1 = 2d_conv(input_images, 256, kernel-size=5)
# layer2 = 2d_conv_slim_capsule(layer1, channels=32, vec_dim=8) # produces a tensor of size [batch, vec_dim, num_channels, height, width] when given input_tensor (a scalar input_tensor, i.e. vec_dim =1 for this tensor)
# layer2b = _routing(layer2, layer_dimensions=[batch, vec_dim, num_channels, height, width], squash_biases, num_routing)
# layer3 = capsule_layer(layer2b, apply_weights=True)
# layer3b = _routing(layer3, layer_dimensions=[batch, vec_dim=16, num_channels=10, height=1, width=1], squash_biases, num_routing)
# layer4 = logistic_fc_neural_network(flatten(layer3b))
# output = layer4


def _patch_based_routing(input_tensor, output_dimensions=None, squash_biases=None, num_routing=3, patch_size=3, patch_stride=1):
    raise NotImplemented

def _routing(input_tensor, output_dimensions=None, squash_biases=None, num_routing=3, bias_channel_sharing=False):
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


    input_tensor_shape = tf.shape(input_tensor)
    logit_shape_input = input_tensor_shape[:]
    logit_shape_input[1] = 1
    logit_shape = logit_shape_input[:] + output_dimensions
    # logit_shape = [batch, 1, num_channels, height, width,  num_channels_output, height_output, width_output]

    reduced_logit_shape = logit_shape_input + [np.product(output_dimensions)]

    vec_dim = input_tensor_shape[1]

    if(squash_biases === None):
        bias_dimensions = output_dimensions[:]
        if(bias_channel_sharing===True): # wish to share bias terms across all channels
            bias_dimensions[0] = 1
            squash_biases = variables.new_bias_variable(shape=bias_dimensions)
            tiling_term = [1] * tf.size(output_dimensions)
            tiling_term[0] = output_dimensions[0]
            squash_biases = tf.tile(squash_biases, tiling_term)
        else: # create a unique bias term for each output capsule across all channels (and width and height)
            squash_biases = variables.new_bias_variable(shape=output_dimensions)

    def _algorithm(i, logits, voting_tensors):
        logits = tf.reshape(logits, shape=reduced_logit_shape)
        c_i = softmax(logits, dim=length(reduced_logit_shape-1))
        # c_i = [batch, 1, num_channels, height, width, total_num_of_output capsules = product(num_channels_output, height_output, width_output)]
        #input_tensor shape = [batch, vec_dim, num_channels, height, width]

        c_i = tf.reshape(c_i, shape=logit_shape)

        tmp = [1]*len(reduced_logit_shape)
        tmp[1] = vec_dim
        c_i = tf.tile(c_i, tmp)

        tmp = [1]*len(input_tensor_shape)
        tmp = tmp[:] + output_dimensions
        tmp_tensor = tf.tile(input_tensor, tmp) # dimensoins: [batch, vec_dim, num_channels, height, width,,,, num_channels_output, height_output, width_output]

        s_j = tf.multiply(c_i, tmp_tensor, name="Individual Voting")
        dim_to_sum = range(2, len(input_tensor_shape))
        s_j = tf.reduce_sum(s_j, axis=dim_to_sum, name="Voting Results")
        v_j = _squash(s_j, name="Non-linearity of votes")
        #v_j dimensions = [batch, vec_dim, num_channels_output, height_output, width_output]


        tmp = [1]*len(input_tensor_shape)
        tmp = tmp[:] + output_dimensions
        tmp_input_tensor = tf.tile(input_tensor, tmp) # dimensoins: [batch, vec_dim, num_channels, height, width,,,, num_channels_output, height_output, width_output]

        tmp = [1] * len(tf.shape(v_j))
        tmp = tmp[:] + input_tensor_shape[2::]
        tmp_vj_tensor = tf.tile(v_j, tmp) # dimensions: [batch, vec_dim, num_channels_output, height_output, width_output, num_channels, height, width]
        tmp_transpose_list = range(len(tf.shape(tmp_tensor2)))
        tmp_last_elements = tmp_transpose_list[len(tf.shape(vj))::]
        tmp_first_elements = tmp_transpose_list[2:len(tf.shape(vj))]
        tmp_vj_transposer = tmp_last_elements[:] + tmp_first_elements[:]
        tmp_vj_tensor = tf.transpose(tmp_tensor2, tmp_vj_transposer) # dimensoins: [batch, vec_dim, num_channels, height, width,,,, num_channels_output, height_output, width_output]

        dot_product = tf.multiply(tmp_input_tensor, tmp_vj_tensor, name="'Cosine' dot product: multiply")
        # dimensions of dot product: [batch, vec_dim, num_channels, height, width,,,, num_channels_output, height_output, width_output]
        dot_product = tf.reduced_sum(dot_product, axis=[1], keepdims=True,  name="'Cosine' dot product: sum")
        # dimensions of dot product: [batch,    1   , num_channels, height, width,,,, num_channels_output, height_output, width_output]


        # logit_shape = [batch, 1, num_channels, height, width,  num_channels_output, height_output, width_output]
        logits = tf.add(logits, dot_product, name="b_ij logit update")

        # [] [unfinished] - algorithm only works when the output_dimensions has the same parameters as the input_tensors, i.e:
        # no change in the num_channels, width or height. this is because of the 'dot product' step present in the algorithm

        voting_tensors.write(i, v_j)

        return (i+1, logits, voting_tensors)

    logits = tf.fill(logit_shape, 0., name="b_ij")
    voting_tensors = tf.TensorArray(size=num_routing, dtype=tf.float32, clear_after_read=clear_after_read_for_votes) # do not clear after read as we want to investigate
    i = tf.constant(0, dtype=tf.int16)
    _, logits, voting_tensors = tf.while_loop(
        lambda i, logits, voting_tensors: i < num_routing,
        _algorithm,
        loop_vars = [i, logits, voting_tensors],
        swap_memory=gpu_cpu_swap
    )

    return voting_tensors.read(num_routing-1)
