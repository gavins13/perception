import tensorflow as tf
from routing import *
import numpy as np
import variables

def convolutional_capsule_layer_v2(input_tensor, kernel_height, kernel_width, scope_name,output_kernel_vec_dim=8, strides=[1, 1], num_output_channels=None, type="SAME", num_routing=3, use_matrix_bias=True, use_squash_bias=True, supplied_squash_biases=None, squash_He=False, squash_relu=False, convolve=False):
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
        patches_pre_tiling = patches 
        patches = tf.expand_dims(patches, 6)# [M, k_h, k_w, p,  |T^l|, v_d^l, 1]
        patches = tf.expand_dims(patches, 5)# [M, k_h, k_w, p,  |T^l|, 1, v_d^l, 1]
        print(patches.get_shape().as_list())
        patches = tf.tile(patches, [1,1,1,1,1,num_output_channels,1,1])# [M, k_h, k_w, p,  |T^l|, |T^l+1|, v_d^l, 1]
        print(patches.get_shape().as_list())
        patches_shape = patches.get_shape().as_list()

        if(convolve==False):
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
        else:
          # Instead, use convolution to generate the output
          patches = patches_pre_tiling #  # [M, k_h, k_w, p,  |T^l|, v_d^l]
          # we need to produce # [M, k_h, k_w, p,  |T^l|, |T^l+1|,  v_d^l+1, 1]
          # i.e. 3
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
        if(supplied_squash_biases==None):
            print(">>>>>Create squash terms")
            '''squash_bias_shape = prerouted_output_shape2[1:3] + [1, 1] # [ v_d^l+1, |T^l+1|, 1, 1]'''
            squash_bias_shape = prerouted_output_shape2[1:3] + [prerouted_output_shape2[3], 1] # [ v_d^l+1, |T^l+1|, x'*y', 1]
            print(squash_bias_shape)
            if(use_squash_bias==True):
                with tf.variable_scope(scope_name):
                    if(squash_He==False):
                        squash_biases = variables.bias_variable(squash_bias_shape)
                    else:
                        with tf.variable_scope('squash'):
                            squash_biases = variables.weight_variable(squash_bias_shape,He=True, He_nl=np.int(np.prod(squash_bias_shape)))
            else:
                    squash_biases = tf.fill(squash_bias_shape, 0.)
            print([1, 1, prerouted_output_shape2[3], 1])
            '''squash_biases = tf.tile(squash_biases, [1, 1, prerouted_output_shape2[3], 1])'''
            # [v_d^l+1, |T^l+1|, x'*y', 1]
            '''squash_bias_shape = prerouted_output_shape2[1:3] + [ prerouted_output_shape2[3], 1] # [ v_d^l+1, |T^l+1|, x'*y', 1]
            squash_biases = tf.fill(squash_bias_shape, 0.)'''
            # [v_d^l+1, |T^l+1|, x'*y', 1]
        else:
            squash_biases = supplied_squash_biases
            squash_bias_shape = squash_biases.get_shape().as_list()


        print(">>>>> Perform routing")
        patch_shape=[1,1,prerouted_output_shape[-1]]
        print(patch_shape)
        #routed_output = patch_based_routing(prerouted_output, scope_name+'/routing', squash_biases=squash_biases,  num_routing=num_routing, patch_shape=patch_shape, patch_stride=[1,1,1],deconvolution_factors=None, bias_channel_sharing=False)
        routed_output = patch_based_routing_for_convcaps(prerouted_output, squash_biases=squash_biases,  num_routing=num_routing, squash_relu=squash_relu)

        print(">>>>> Finished Routing")
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
