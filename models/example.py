import sys

import tensorflow as tf
import numpy as np
import collections

sys.path.insert(0, 'capsules/operations')
from capsule_operations import LocalisedCapsuleLayer


sys.path.insert(0, 'lib/')
from architecture import architecture_base

Results = collections.namedtuple('Results', ('output','k_space_pred', 'diagnostic_1', 'diagnostic_2'))

sys.path.insert(0, 'models/lib')
import variables
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import os

class architecture(architecture_base):
    def __init__(self):
        # Use __set_learning__() to set optimizer
        self.hyperparameters.learning_rate = 1.e-4

    def __test_config__(self):
        pass

    def build(self, input_images, k_space_mask, k_space_masked, k_space_full, add_noise=False):
        self.__test_config__()
        #input images = [batch, height,width] BUT COMPLEX TYPE!
        its = input_images.get_shape().as_list()
        input_images_real = tf.expand_dims(tf.math.real(input_images), axis=3)
        input_images_imag = tf.expand_dims(tf.math.imag(input_images), axis=3)
        input_images= tf.concat([input_images_real, input_images_imag], axis=3) # [batch, height, width, 2]

        k_space_mask = tf.cast(k_space_mask, dtype=tf.complex64)
        k_space_masked = tf.cast(k_space_masked, dtype=tf.complex64)

        if(add_noise==True):
            def gaussian_noise_layer(input_layer, std=100.):
                noise = tf.random.normal(shape=input_layer.get_shape().as_list(), mean=0.0, stddev=std, dtype=tf.float32)
                return input_layer + noise
            input_images = gaussian_noise_layer(input_images)


        def cnn(input_images, nd=5):
            its = input_images.get_shape().as_list()
            print(">>>>> Initial Convolution")
            layer1 = input_images
            vec_to_output_layers = 4
            for i in range(nd-1-vec_to_output_layers):
                layer1 = tf.compat.v1.layers.conv2d(layer1, 64, 3, padding='same', name='Conv'+str(i), activation=tf.nn.relu, kernel_initializer=tf.compat.v1.keras.initializers.he_normal(seed=None))
            layer1 = tf.compat.v1.layers.conv2d(layer1, 32, 3, padding='same', name='ConvCon1'+str(i), activation=tf.nn.relu, kernel_initializer=tf.compat.v1.keras.initializers.he_normal(seed=None))
            layer1 = tf.compat.v1.layers.conv2d(layer1, 16, 3, padding='same', name='ConvCon2'+str(i), activation=tf.nn.relu, kernel_initializer=tf.compat.v1.keras.initializers.he_normal(seed=None))
            def onehot_capsule(layer1):
              output_channels = 4
              normalise = True
              layer1 = tf.transpose(a=tf.reshape(tf.expand_dims(layer1, axis=4), its[0:3] + [4,4]), perm=[0,3,4,1,2])
              CapsLayer = LocalisedCapsuleLayer(k=3, output_vec_dim=4, num_output_channels=output_channels, num_routing=6, use_matrix_bias=True, use_squash_bias=False, squash_He=True, matrix_tanh=False, routing_softmax_opposite=True, squash_relu=False)
              layer1 = CapsLayer(tf.cast(layer1, dtype=tf.float32), 'ConvCaps')  # [M, v_d^l+1, |T^l+1|, x,y]
              layer4 = tf.norm(tensor=layer1, ord='euclidean', axis=1,  keepdims=False)   # [M, |T^l+1|, x,y]
              layer4_ind = tf.argmax(input=tf.transpose(a=layer4, perm=[0,2,3,1]), axis=3) # [M, x,y]
              layer4_mask = tf.one_hot(layer4_ind, depth=output_channels, on_value=1., off_value=0., axis=None) # [M,x,y,T^l+1]
              print("//EndOneHot")
              capschannelvotes = tf.transpose(a=layer4_mask, perm=[1,2,3,0])
              layer4_mask = tf.transpose(a=tf.expand_dims(layer4_mask, axis=4), perm=[0,4,3,1,2]) # [M, 1, T^l+1, x , y]
              layer1_normalised = tf.nn.l2_normalize(layer1, axis=1) if(normalise==True) else layer1
              print("//Norm")
              layer5 = tf.multiply(layer1_normalised, layer4_mask)
              layer6 = tf.squeeze(tf.reshape(tf.transpose(a=layer5, perm=[0,3,4,1,2]), its[0:3]+[-1,1]), axis=4) #  [M,, x, y, v_d^l+1*|T^l+1|]
              return layer6
            layer1 = onehot_capsule(layer1)
            print("//end one hot capsule")
            for i in range(vec_to_output_layers):
                layer1 = tf.compat.v1.layers.conv2d(layer1, 64, 3, padding='same', name='ConvEnd'+str(i), activation=tf.nn.relu, kernel_initializer=tf.compat.v1.keras.initializers.he_normal(seed=None))
            layer2 = tf.compat.v1.layers.conv2d(layer1, 2, 3, padding='same', name='Conv_Recon', activation=tf.nn.relu, kernel_initializer=tf.compat.v1.keras.initializers.he_normal(seed=None))
            return layer2

        def data_consistency(input_images, l):
            def NOT(input):
                #return tf.logical_not(input)
                #return tf.cast(tf.equal(input, 0.), dtype=tf.complex64, name="notgate")
                return tf.cast(tf.subtract(tf.cast(1., dtype=tf.complex64), input), dtype=tf.complex64, name="notgate")

            def weighted_mean(raw, predict):
                #l = init()
                ll_1 = tf.cast(l, dtype=tf.complex64, name="ll_1cast")
                #ll_2 = tf.cast(l+1., dtype=tf.complex64, name="ll_2cast")
                ll_2 = tf.add(l, 1., name="addd")
                ll_2 = tf.cast(ll_2, dtype=tf.complex64, name="ll_2cast")
                ll_3 = tf.math.reciprocal(ll_2, name="recip1")
                predict = tf.multiply(predict, ll_3)
                raw = tf.multiply(raw, tf.divide(ll_1, ll_2) )
                return tf.add(predict, raw)
            input_image_real = tf.squeeze(tf.slice(input_images, [0,0,0,0], its + [1]), axis=3)
            input_image_imag = tf.squeeze(tf.slice(input_images, [0,0,0,1], its + [1]), axis=3)
            input_images = tf.complex(input_image_real, input_image_imag)
            input_images = tf.cast(input_images, dtype=tf.complex64, name="input_images_cast")
            k_space = tf.signal.fft2d(input_images)
            print(">>K-space shape")
            print(k_space.get_shape().as_list())
            print(k_space_mask.get_shape().as_list())
            print(NOT(k_space_mask).get_shape().as_list())
            print(k_space_masked.get_shape().as_list())
            new_k_space = tf.multiply(k_space, NOT(k_space_mask))
            print(">> K-space predictions")
            k_space_predictions = tf.multiply(k_space, k_space_mask)
            print(">> K-space raw values")
            k_space_raw_values = tf.multiply(k_space_masked, k_space_mask) # This line of code is redundant (but here just in case)
            print(">> K-space weighted mean")
            k_space_DC = weighted_mean(k_space_raw_values, k_space_predictions)
            _tmp_image_space_DC = tf.abs(tf.signal.ifft2d(k_space_DC))
            print(">> DC")
            new_k_space = tf.add(new_k_space, k_space_DC)
            new_input_images = tf.signal.ifft2d(new_k_space)
            new_input_images_real = tf.expand_dims(tf.math.real(new_input_images), axis=3)
            new_input_images_imag = tf.expand_dims(tf.math.imag(new_input_images), axis=3)
            new_input_images = tf.concat([new_input_images_real, new_input_images_imag], axis=3)
            return new_input_images, k_space, _tmp_image_space_DC

        images = input_images
        for i in range(3):
            with tf.compat.v1.name_scope('CNN' + str(i)):
                with tf.compat.v1.variable_scope('CNN' + str(i)):
                    images = cnn(images, 8)
                    dc_weight = variables.bias_variable(init=500., shape=[1])
                    #dc_weight = .00000001
                    #dc_weight = 500.
                    images, k_space_pred, _tmp_image_space_DC = data_consistency(images, dc_weight)
        images_real = tf.squeeze(tf.slice(images, [0,0,0,0], its + [1]), axis=3)
        images_imag = tf.squeeze(tf.slice(images, [0,0,0,1], its + [1]), axis=3)
        output_images = tf.complex(images_real, images_imag)
        output_images = tf.cast(output_images, dtype=tf.complex64)
        _tmp_from_k_gt = tf.abs(tf.signal.ifft2d(tf.cast(k_space_full, dtype=tf.complex64)))



        result = Results(output_images, k_space_pred, _tmp_image_space_DC, _tmp_from_k_gt)
        print(">>>>> Graph Built!")


        return result



    def loss_func(self, input_images, ground_truth, validation_input_images, validation_ground_truth, extra_data, validation_extra_data):
        #input_images = tf.expand_dims(input_images, axis=3)
        input_images = tf.expand_dims(extra_data["image_data_complex"], axis=3)
        ground_truth = tf.expand_dims(ground_truth, axis=3)
        print(">>>>>>Shape")
        print(input_images.get_shape().as_list())
        #validation_input_images = tf.expand_dims(validation_input_images, axis=3)
        validation_input_images = tf.expand_dims(validation_extra_data["image_data_complex"], axis=3)
        validation_ground_truth = tf.expand_dims(validation_ground_truth, axis=3)
        mini_batch_size = input_images.get_shape().as_list()[0]
        validation_mini_batch_size = validation_input_images.get_shape().as_list()[0]


        input_images = tf.squeeze(input_images, axis=3)
        ground_truth = tf.squeeze(ground_truth, axis=3)
        validation_input_images = tf.squeeze(validation_input_images, axis=3)
        validation_ground_truth = tf.squeeze(validation_ground_truth, axis=3)

        k_space_mask = extra_data["k_space_mask"]
        k_space_masked = extra_data["k_space_masked"]
        k_space_full = extra_data["k_space_full"]

        validation_k_space_mask = validation_extra_data["k_space_mask"]
        validation_k_space_masked = validation_extra_data["k_space_masked"]
        validation_k_space_full = validation_extra_data["k_space_full"]


        print(">>>Start Building Architecture.")
        res = self.build(input_images, k_space_mask, k_space_masked, k_space_full)
        print(">>>Finished Building Architecture.")
        output = res.output
        output = tf.abs(output)
        def find_norm_new(complex_input):
            real_part = tf.math.real(complex_input)
            imag_part = tf.math.imag(complex_input)
            real = tf.pow(real_part, 2.)
            imag = tf.pow(imag_part, 2.)
            end = tf.add(real, imag)
            final = tf.reduce_sum(input_tensor=end, axis=[1,2], keepdims=False)
            return final
        def find_norm(input):
            #return tf.norm(input, axis=[1,2])
            input = tf.pow(tf.abs(input), 2)
            return tf.reduce_sum(input_tensor=input, axis=[1,2])

        print(">>> Run on validation set")
        validation_res = self.build(validation_input_images, validation_k_space_mask, validation_k_space_masked, validation_k_space_full)
        validation_output = validation_res.output
        validation_output = tf.abs(validation_output)
        print(">>> Find MSE for the validation set")
        v_diff = tf.subtract(tf.cast(validation_ground_truth, tf.float32), validation_output)
        v_MSE_loss = find_norm(v_diff)
        with tf.compat.v1.name_scope('validation'):
            tf.compat.v1.summary.scalar("validation_total_loss", tf.reduce_sum(input_tensor=v_MSE_loss))
        print(">>>Some Maths on result")
        print(">>>> Find Difference")
        difference = tf.subtract(tf.cast(ground_truth, tf.complex64), tf.cast(res.output, tf.complex64))
        print(">>>> Find Norm")
        #L2_norm = tf.norm(difference, axis=[1,2])
        #L1_norm = tf.abs(difference)
        #L2_norm = L1_norm
        print(">>>> Find Mean of Norm")

        L2_norm = find_norm(difference)
        batch_loss = tf.reduce_sum(input_tensor=L2_norm)
        difference=tf.math.real(difference)
        print(">>>> Find + and - loss")
        positive_loss =  tf.reduce_sum(input_tensor=tf.boolean_mask(tensor=difference, mask=tf.greater(difference, 0.)))
        negative_loss =  tf.reduce_sum(input_tensor=tf.boolean_mask(tensor=difference, mask=tf.less(difference, 0.)))

        print(">>>> PSNR and SSIM")
        psnr = tf.image.psnr(tf.expand_dims(ground_truth, axis=3), tf.expand_dims(tf.abs(output), axis=3), max_val=1114) #1114 3480
        ssim = tf.image.ssim(tf.expand_dims(ground_truth, axis=3), tf.expand_dims(tf.abs(output), axis=3), max_val=1114) #1114 3480

        print(">>>> PSNR Stats")
        max_psnr = tf.reduce_max(input_tensor=psnr)
        min_psnr = tf.reduce_min(input_tensor=psnr)
        mean_psnr = tf.reduce_mean(input_tensor=psnr)


        print(">>>> SSIM Stats")
        max_ssim = tf.reduce_max(input_tensor=ssim)
        min_ssim = tf.reduce_min(input_tensor=ssim)
        mean_ssim = tf.reduce_mean(input_tensor=ssim)


        print(">>>> Find Mean Loss")
        with tf.compat.v1.name_scope('total'):
            print(">>>>>> Add to collection")
            tf.compat.v1.add_to_collection('losses', batch_loss)
            print(">>>>>> Creating summary")
            tf.compat.v1.summary.scalar(name='batch_L2_reconstruction_cost', tensor=batch_loss)
            print(">>>> Add result to collection of loss results for this tower")
            all_losses = tf.compat.v1.get_collection('losses') # [] , this_tower_scope) # list of tensors returned
            total_loss = tf.add_n(all_losses) # element-wise addition of the list of tensors
            #print(total_loss.get_shape().as_list())
            tf.compat.v1.summary.scalar('total_loss', total_loss)
        print(">>>> Add results to output")
        with tf.compat.v1.name_scope('accuracy'):
            tf.compat.v1.summary.scalar('max_psnr', max_psnr)
            tf.compat.v1.summary.scalar('min_psnr', min_psnr)
            tf.compat.v1.summary.scalar('mean_psnr', mean_psnr)
            tf.compat.v1.summary.scalar('max_ssim', max_ssim)
            tf.compat.v1.summary.scalar('min_ssim', min_ssim)
            tf.compat.v1.summary.scalar('mean_ssim', mean_ssim)
            tf.compat.v1.summary.scalar('positive_loss', positive_loss)
            tf.compat.v1.summary.scalar('negative_loss', tf.multiply(negative_loss, -1.))
            model_output_f1 = tf.expand_dims(tf.slice(tf.cast(tf.abs(output), dtype=tf.float64), [0,0,0], [1, -1,-1]), axis=3)
            model_input_f1 = tf.expand_dims(tf.slice(tf.cast(tf.abs(input_images), dtype=tf.float64), [0,0,0], [1, -1,-1]), axis=3)
            model_gt_f1 = tf.expand_dims(tf.slice(tf.cast(ground_truth, dtype=tf.float64), [0,0,0], [1, -1,-1]), axis=3)
            tf.compat.v1.summary.image('model_output',  model_output_f1)
            tf.compat.v1.summary.image('model_input',  model_input_f1)
            tf.compat.v1.summary.image('model_ground_truth',  model_gt_f1)
            tf.compat.v1.summary.image('model_diff_gt_output',  model_gt_f1 - model_output_f1)
            tf.compat.v1.summary.image('model_diff_input_output',  model_input_f1 - model_output_f1)
            tf.compat.v1.summary.image('diagnostic_1', tf.expand_dims(tf.slice(res.diagnostic_1, [0,0,0], [1, -1,-1]), axis=3))
            tf.compat.v1.summary.image('diagnostic_2', tf.expand_dims(tf.slice(res.diagnostic_2, [0,0,0], [1, -1,-1]), axis=3))



        print(">>>> Add results to output (validation)")
        with tf.compat.v1.name_scope('validation'):
            tf.compat.v1.summary.image('input', tf.expand_dims(tf.abs(validation_input_images), axis=3))
            tf.compat.v1.summary.image('ground_truth', tf.expand_dims(validation_ground_truth, axis=3))
            tf.compat.v1.summary.image('output', tf.expand_dims(tf.abs(validation_output), axis=3))
            tf.compat.v1.summary.scalar('psnr', tf.reduce_mean( input_tensor=tf.image.psnr(tf.expand_dims(validation_ground_truth, axis=3), tf.abs(tf.expand_dims(validation_output, axis=3)), max_val=1114)))
            tf.compat.v1.summary.scalar('ssim', tf.reduce_mean( input_tensor=tf.image.ssim(tf.expand_dims(validation_ground_truth, axis=3), tf.abs(tf.expand_dims(validation_output, axis=3)), max_val=1114)))
            tf.compat.v1.summary.image('k_space_real_gt', tf.expand_dims(tf.slice(tf.math.real(validation_k_space_masked), [0,0,0], [1, -1,-1]), axis=3))
            tf.compat.v1.summary.image('k_space_image_gt', tf.expand_dims(tf.slice(tf.math.imag(validation_k_space_masked), [0,0,0], [1, -1,-1]), axis=3))
            tf.compat.v1.summary.image('k_space_real', tf.expand_dims(tf.slice(tf.math.real(validation_res.k_space_pred), [0,0,0], [1, -1,-1]), axis=3))
            tf.compat.v1.summary.image('k_space_image', tf.expand_dims(tf.slice(tf.math.imag(validation_res.k_space_pred), [0,0,0], [1, -1,-1]), axis=3))
            tf.compat.v1.summary.image('k_space_mask', tf.expand_dims(tf.slice(tf.math.real(validation_k_space_mask), [0,0,0], [1, -1,-1]), axis=3))
            tf.compat.v1.summary.image('diagnostic_1', tf.expand_dims(tf.slice(validation_res.diagnostic_1, [0,0,0], [1, -1,-1]), axis=3))
            tf.compat.v1.summary.image('diagnostic_2', tf.expand_dims(tf.slice(validation_res.diagnostic_2, [0,0,0], [1, -1,-1]), axis=3))

        #squashes = tf.get_variable('ConvCaps1/biases',   None,    dtype=tf.float32, trainable=True)

        diagnostics = {'max_psnr': max_psnr, 'min_psnr': min_psnr, 'mean_psnr': mean_psnr, 'max_ssim':max_ssim, 'min_ssim':min_ssim, 'mean_ssim':mean_ssim, 'positive_loss':positive_loss, 'negative_loss':negative_loss, 'total_loss':total_loss, 'psnr': psnr, 'ssim': ssim, 'mse': L2_norm}
        return output, batch_loss, diagnostics, [] #, [tf.get_variable('ConvCaps1/squash/weights')]

    def analyse(self, main_results, full_diagnostics, user_diagnostics, save_directory):
        diag = user_diagnostics
        path_orig = save_directory + '/original/'
        os.mkdir(path_orig, 0o755)
        path_mri = save_directory + '/mri/'
        os.mkdir(path_mri, 0o755)
        main_results = {"x": [], "y": [], "gt": []}
        for i in range(len(main_results["y"])):
            print("Saving results %s of %s" % (i, len(main_results["y"])))
            print(np.squeeze(main_results["x"][i]).shape)
            print(np.squeeze(main_results["y"][i]).shape)
            print(np.squeeze(main_results["gt"][i]).shape)
            fig=plt.figure(figsize=(10,8), dpi=80, facecolor='w', edgecolor='k')
            print("fig diag")
            fig.suptitle("PSNR: " + str(diag['psnr'][i]) + "; SSIM: " + str(diag['ssim'][i]) + "; MSE: " + str(diag['mse'][i]), fontsize=8)
            print("data to fig")
            plt.subplot(231)
            plt.gca().set_title('x')
            #x = skimage.transform.resize(np.squeeze(input_datas[i])[64:192, 64:192], [128,128],preserve_range=True)
            x = np.squeeze(main_results["x"][i])
            plt.imshow(np.abs(x), cmap=plt.cm.gray)
            #plt.colorbar()
            print("data to fig")
            plt.subplot(232)
            plt.gca().set_title('y')
            plt.imshow(np.squeeze(main_results["y"][i]), cmap=plt.cm.gray)
            print("data to fig")
            plt.subplot(233)
            plt.gca().set_title('gt')
            #gt=skimage.transform.resize(np.squeeze(ground_truths[i])[64:192, 64:192], [128,128],preserve_range=True)
            gt=np.squeeze(main_results["gt"][i])
            plt.imshow(gt, cmap=plt.cm.gray)
            print("data to fig")
            plt.subplot(234)
            plt.gca().set_title('gt-y')
            plt.imshow(gt-np.squeeze(main_results["y"][i]), cmap=plt.cm.gray)
            plt.subplot(235)
            plt.gca().set_title('y-x')
            plt.imshow(np.squeeze(main_results["y"][i]) - np.abs(x), cmap=plt.cm.gray)
            plt.subplot(236)
            plt.gca().set_title('gt-x')
            plt.imshow(gt - np.abs(x), cmap=plt.cm.gray)
            plt.savefig(path_mri + str(i) + ".png")
            plt.close(fig)
        zipped_diag = list(zip(diag['psnr'], diag['ssim'], diag['mse']))
        with open(save_directory + '/diagnostics.csv', 'w') as the_file:
            the_file.write('psnr, ssim, mse\n')
            for entry in zipped_diag:
                for val in list(entry):
                    the_file.write(str(val))
                    the_file.write(',')
                the_file.write('\n')
