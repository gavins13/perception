import tensorflow as tf
from routing import *
import numpy as np
import variables
from functools import reduce

class LocalisedCapsuleLayer(object): # LocalCaps
    def __init__(self, k=3, kernel_height=None, kernel_width=None, output_vec_dim=8, strides=[1,1], num_output_channels=8, type="SAME", num_routing=3, use_matrix_bias=True, use_squash_bias=True, supplied_squash_biases=None, squash_He=False, squash_relu=False, convolve=False, matrix_tanh=False, routing_softmax_opposite=False):
        kernel_height = k if(kernel_height==None) else kernel_height
        kernel_width =  k if(kernel_width== None) else kernel_width
        self.params = {"k_h": kernel_height, "k_w": kernel_width, "strides": strides, "num_routing": num_routing}
        self.output = {"channels": num_output_channels, "vec_dim": output_vec_dim, 'x': None, 'y': None}
        self.options= {"type": type, "use_matrix_bias": use_matrix_bias, "use_squash_bias": use_squash_bias, "supplied_squash_biases": supplied_squash_biases, "squash_He": squash_He,"squash_relu": squash_relu, "convolve": convolve, "matrix_tanh": matrix_tanh, "routing_softmax_opposite": routing_softmax_opposite}
        self.scope_name='LocalCaps'
    def transform(self, input_tensor):  # [batch, vec_dim, num_ch, h, w] ---> [M, x, y, v_d^l+1, |T^l|, |T^l+1|]
        input_tensor_shape = input_tensor.get_shape().as_list()
        its = input_tensor_shape
        self.output["x"], self.output["y"] = its[3], its[4]
        print("input tensor shape")
        print(input_tensor_shape)
        # [batch, vec_dim, num_ch, h, w]
        if(self.options["convolve"]==False):
          matrix_shape = [1,self.output["channels"],   its[2],1,1, self.output["vec_dim"], its[1]]
          matrix_bias_shape = [1,self.output["channels"], its[2], 1,1, self.output["vec_dim"], 1]
          tmp = tf.expand_dims(tf.expand_dims(input_tensor, axis=5), axis=6)
          tmp = tf.transpose(tmp, [0,6, 2,3,4,1, 5]) # [M, 1, T^l, x, y, v_d, 1]
          with tf.variable_scope(self.scope_name):
              with tf.variable_scope('matrix_transform'):
                  if(self.options["use_matrix_bias"]==True):
                      matrix_bias = variables.bias_variable(matrix_bias_shape)
                  matrix = variables.weight_variable(matrix_shape)
          if(self.options["matrix_tanh"] == True):
              matrix = tf.nn.tanh(matrix)
          votes = tf.matmul(tf.tile(matrix, [its[0],1,1,self.output["x"],self.output["y"],1,1]), tf.tile(tmp, [1,self.output["channels"],1,1,1,1,1]))
          votes = tf.add(votes, matrix_bias) if(self.options["use_matrix_bias"]==True) else votes
          # [M, T^l+1, T^l, x,y, v_d+1, 1]
          votes = tf.transpose(tf.squeeze(votes,axis=6), [0, 3, 4, 5, 2, 1]) # [M, x, y, v_d^l+1, |T^l|, |T^l+1|]
        else:
          raise NotImplementedError()
          '''for input_channel_number in range(its[2]):
              tmp = its[:]
              tmp[2] =1
              this_channel = tf.slice(input_tensor, [0,0,input_channel_number,0,0], tmp) # [batch, vec_dim, 1(a channel), h, w]
              this_channel = tf.squeeze(tf.transpose(this_channel, [0,3,4,1,2]), axis=4) # [batch, x, y, vec_dim]
              for conv_it in range(1):
                  this_channel = tf.layers.conv2d(this_channel, its[1], k, padding='same', activation)'''
        return votes
    def localise(self, input_tensor): # [M, x, y, v_d^l+1, |T^l|, |T^l+1|] ---> [M,v_d^l+1, |T^l+1|,p, k_h*k_w*|T^l|]
        votes_shape = input_tensor.get_shape().as_list()
        # new input: [M, x, y, v_d^l+1, |T^l|, |T^l+1|]
        patches = tf.transpose(input_tensor, [1,2,3,4,5,0]) #  [x, y, v_d^l+1, |T^l|, |T^l+1|, M]
        patches_shape = patches.get_shape().as_list()
        patches = tf.reshape(patches, patches_shape[0:2] + [reduce(lambda x,y: x*y, patches_shape[2:6]), 1,1,1])     # [x, y, v_d^l+1* |T^l|* |T^l+1|* M, 1, 1, 1]
        patches = tf.squeeze(patches, axis=[4,5]) # [x, y, v_d^l+1* |T^l|* |T^l+1|* M, 1]
        patches = tf.transpose(patches, [2,0,1,3])  #  [v_d^l+1* |T^l|* |T^l+1|* M, x, y, 1]

        patches = tf.extract_image_patches(patches, [1,self.params["k_h"], self.params["k_w"], 1], strides=[1]+self.params["strides"]+[1], rates=[1,1,1,1], padding=self.options["type"]) #  [ v_d^l+1|T^l|*|T^l+1|*M, x, y, k_w*k_h]
        patches_new_shape = patches.get_shape().as_list()
        self.output["x"], self.output["y"] = patches_new_shape[1:3]

        patches = tf.expand_dims(patches, axis=4)
        patches = tf.reshape(patches, patches_new_shape[0:3] + [self.params["k_h"], self.params["k_w"]])   #  [ v_d^l+1|T^l|*|T^l+1|*M, x, y, k_w, k_h]
        patches = tf.transpose(patches, [1,2,3,4,0])
        patches = tf.expand_dims(tf.expand_dims(tf.expand_dims(patches, axis=5), axis=6), axis=7)
        patches = tf.reshape(patches, patches_new_shape[1:3] + [self.params["k_h"], self.params["k_w"]] + patches_shape[2:6]) # [x, y, k_w, k_h, v_d^l+1, |T^l|, |T^l+1|, M]
        patches_main_shape = patches.get_shape().as_list()
        patches = tf.transpose(patches, [7, 4, 6, 0,1, 2,3,5])
        patches_new_shape = patches.get_shape().as_list() # [M, v_d^l+1, |T^l+1|, x,y, k_h, k_w, |T^l|]
        patches = tf.squeeze(tf.reshape(patches, patches_new_shape[0:5] + [reduce(lambda x,y: x*y, patches_new_shape[5:8]), 1, 1]), axis=[6,7])   # [M, v_d^l+1, |T^l+1|, x,y, k_h* k_w*|T^l|]
        patches = tf.transpose(patches, [0,1,2,5,3,4])
        patches_new_shape = patches.get_shape().as_list()
        patches = tf.squeeze(tf.reshape(patches, patches_new_shape[0:4]+[patches_new_shape[4]*patches_new_shape[5], 1]), axis=5) # [M, v_d^l+1, |T^l+1|, k_h* k_w*|T^l|, x*y]
        patches = tf.transpose(patches, [0,1,2,4,3])
        # GOAL : # [M,v_d^l+1, |T^l+1|,p, k_h*k_w*|T^l|]
        return patches
    def route(self, votes):# Votes has shape:  [M, v_d^l+1, |T^l+1|, x'*y', VOTES] ---->  [M, v_d^l+1, |T^l+1|, x', y']
        # Output needs to be: [M, v_d^l+1, |T^l+1|, x', y'], i.e. need to remove the votes
        if(self.options["supplied_squash_biases"]==None):
          print(">>>>>Create squash terms")
          squash_bias_shape = [self.output["vec_dim"], self.output["channels"], self.output["x"]*self.output["y"], 1] # [ v_d^l+1, |T^l+1|, x'*y', 1]
          print(squash_bias_shape)
          if(self.options["use_squash_bias"]==True):
              with tf.variable_scope(self.scope_name):
                  with tf.variable_scope('squash'):
                      squash_biases = variables.bias_variable(squash_bias_shape) if(self.options["squash_He"]==False) else variables.weight_variable(squash_bias_shape,He=True, He_nl=np.int(np.prod(squash_bias_shape)))
          else:
              squash_biases = False
          # [v_d^l+1, |T^l+1|, x'*y', 1]
        else:
          squash_biases = self.options["supplied_squash_biases"]
          squash_bias_shape = squash_biases.get_shape().as_list()


        print(">>>>> Perform routing")
        routed_output = route_votes(votes, squash_biases=squash_biases,  num_routing=self.params["num_routing"], squash_relu=self.options["squash_relu"], softmax_opposite=self.options["routing_softmax_opposite"], use_squash_bias=self.options["use_squash_bias"])
        print(">>>>> Finished Routing")
        # M, v_d^l+1, |T^l+1|, x'*y', 1
        routed_output_shape = routed_output.get_shape().as_list()
        _end = [self.output["x"],self.output["y"]]
        output_tensor = tf.reshape(routed_output, routed_output_shape[0:3] + _end)
        # M, v_d^l+1, |T^l+1|, x', y'
        return output_tensor
    def __call__(self, input_tensor, scope_name):
        self.scope_name = scope_name
        print(">>>> %s START" % scope_name)
        with tf.name_scope(scope_name):
            transform_caps = self.transform(input_tensor)
            local_votes = self.localise(transform_caps)
            output = self.route(local_votes)
        print(output.get_shape().as_list())
        print(">>>> %s END" % scope_name)
        return output
