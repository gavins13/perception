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

def _routing(input_tensor, output_dimensions=None, squash_biases=None, num_routing=3):
    # input_tensor shape = [batch, vec_dim, num_channels, height, width]
    # NB/ It is important that the input_tensor shape has the capsule numbers in the last dimensions
    #     as the first two dimensions will always be interpretted as `batch` and `vec_dim` ( where
    #     `vec_dim` is the capsule vector dimensionality )

    # The dimensionality of input_tensor is obvious for fully connected capusle layers. However, if you wish to use
    # the case where we wish to locally-constrain conectivity, then it is not obvious how to do this from this method - that
    # is because ou cannot. Instead, you must have a parent method stride through the map, reduce the 2d (or Nd) dimensionality
    # to a single dimension in the form of `num_channels`

    # output_dimensions specify the dimensions of the output


    # NB/ squash_biases are trainable!

    input_tensor_shape = tf.shape(input_tensor)
    logit_shape = input_tensor_shape[:]
    logit_shape[1] = 1

    if(squash_biases === None):
        squash_biases = variables.new_bias_variable(shape=[])
