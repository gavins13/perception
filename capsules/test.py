from operations.capsule_operations import *
impor tensorflow as tf
#input images = [batch, height,width]

layer1 = init_conv_2d(input_images, 256, kernel-size=9)
layer2 = convolutional_capsule_layer(layer1, 9,9,output_kernel_vec_dim=8,convolve_across_channels=True,    num_output_channels=32, kernel_is_vector=False)
# layer2b = _routing(layer2, layer_dimensions=[batch, vec_dim, num_channels, height, width], squash_biases, num_routing)
layer2b = _patch_based_routing(layer2, patch_shape=[1,5,5])
layer3 = fc_capsule_layer(layer2b, "Layer3", apply_weights=True,    share_weights_within_channel=False, output_vec_dimension=16)
layer3b = _routing(layer3, output_dimensions=[10,1,1],    bias_channel_sharing=True,) # [batch, 16, num_ch, height, width] = [b,16,10,1,1]
#layer3b = _patch_based_routing(layer3, patch_shape=None)  32 channel output
#layer4 = logistic_fc_neural_network(flatten(layer3b))
layer4 = tf.squeeze(layer3b, axis=[3,4])
layer4 = tf.tranpose(layer4, [0,2,1])
output = tf.norm(layer4, axis=2)
