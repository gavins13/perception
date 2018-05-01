# Usage

 when defining the network, we will write it out like this:
 * layer1 = 2d_conv(input_images, 256, kernel-size=5)
 * layer2 = convolutional_capsule_layer(layer1, channels=32, vec_dim=8) # produces a tensor of size [batch, vec_dim, num_channels, height, width] * when given input_tensor (a scalar input_tensor, i.e. vec_dim =1 for this tensor)
 * layer2b = _routing(layer2, layer_dimensions=[batch, vec_dim, num_channels, height, width], squash_biases, num_routing)
 * layer3 = capsule_layer(layer2b, apply_weights=True, share_weights_within_channel=False,  output_vec_dimension=16)
 * layer3b = _routing(layer3, layer_dimensions=[batch, vec_dim=16, num_channels=10, height=1, width=1], squash_biases, num_routing)
 * layer4 = logistic_fc_neural_network(flatten(layer3b))
 output = layer4

# Key Notes
* The capsule "channel_number" frequently referered to here is the same as the `capsule_type` from the LaLonde et al paper


#  Questions

 1. Checking the routing algorithm. :
   - Currently, in accordance with the Sabour 2017 code, the bias term is learned
   - In the Sabour code, the bias term is shared across channels
   - Why is the bias term learnt instead of just learning the initialisation of the logits?
