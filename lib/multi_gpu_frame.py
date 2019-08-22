import tensorflow as tf
import numpy as np
import sys

import collections
TowerResult = collections.namedtuple('TowerResult', ('result', 'grads', 'loss', 'diagnostics', 'ground_truth', 'input_data'))
JoinedResult = collections.namedtuple('JoinedResult', ('summary', 'train_op','total_loss', 'diagnostics', 'full_diagnostics'))

sys.path.insert(0, 'lib/')
from learning_core import learning_core

class multi_gpu_model(learning_core):
  def __init__(self, ArchitectureObject=None, cpu_only=False, eager=False):
    self.ArchitectureObject = None
    self.cpu_only = cpu_only
    self.eager=eager
    if(ArchitectureObject is not None):
      self.strap_architecture(ArchitectureObject)
    else:
      print(" ... Time to load architecture ... ")

  def run_multi_gpu(self, DataObject, num_gpus=1):
    DataObject.set_num_gpus(num_gpus)
    print(">>>>Using %d GPUs" % num_gpus)
    if(self.ArchitectureObject is None):
        raise Exception('problem with architecture: not loaded')
    tower_grads = []
    results = []
    losses = []
    diagnostics = []
    ground_truths = []
    input_data = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(num_gpus):
        print('>>Assignment of data to tower/GPU %d' % i)
        input_data_shape, input_labels_shape = DataObject.get_data_shape(gpu=i)
        validation_input_data_shape, validation_input_labels_shape = DataObject.get_validation_data_shape(gpu=i)
        print('>>>Data Shapes')
        print(input_data_shape)
        print(input_labels_shape)
        tower_output = self._single_tower(i, input_data_shape, input_labels_shape,  validation_input_data_shape, validation_input_labels_shape)

        print(">>>Grad shapes")
        results.append(tower_output.result)
        tower_grads.append(tower_output.grads)
        losses.append(tower_output.loss)
        diagnostics.append(tower_output.diagnostics)
        ground_truths.append(tower_output.ground_truth)
        input_data.append(tower_output.input_data)
    print('>> Sumarise results from all towers')
    summarized_results = self._summarize_towers(tower_grads, losses, diagnostics)
    print('>> Return results from all towers')
    return summarized_results, results, ground_truths, input_data

  def _single_tower(self, tower_ind, input_data_shape, input_labels_shape, validation_input_data_shape, validation_input_labels_shape, num_targets=1):
    if(self.ArchitectureObject is None):
        raise Exception('problem with architecture: not loaded')

    if(self.cpu_only==True):
        device_name_prefix = 'cpu'
    else:
        device_name_prefix = 'gpu'

    with tf.device('/' + device_name_prefix + ':%d' % tower_ind):
      input_data = tf.placeholder(tf.float32, shape=input_data_shape, name="InputDataGPU"+str(tower_ind))
      input_labels = tf.placeholder(tf.float32, shape=input_labels_shape, name="InputLabelsGPU"+str(tower_ind))

      validation_input_data = tf.placeholder(tf.float32, shape=input_data_shape, name="ValidationInputDataGPU"+str(tower_ind))
      validation_input_labels = tf.placeholder(tf.float32, shape=input_labels_shape, name="ValidationInputLabelsGPU"+str(tower_ind))
      with tf.name_scope('tower_%d' % (tower_ind)) as scope:


        tf.get_variable_scope().reuse_variables()
        if(self.eager==False):
          with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
            print(input_data.get_shape().as_list())
            input_labels = tf.convert_to_tensor(input_labels, dtype=tf.float32)
            output, loss, diagnostics = self.ArchitectureObject.loss_func(input_images=input_data, ground_truth=input_labels, validation_input_images=validation_input_data, validation_ground_truth=validation_input_labels)
            grads_and_vars  = self._optimizer.compute_gradients(loss) # [] [unfinished] why
        else:
            #grads_and_vars  = self._optimizer.compute_gradients(losses_func)
            #print(">>> Use contrib.eager gradient finder")
            grad_function = tf.contrib.eager.implicit_value_and_gradients(self.ArchitectureObject.loss_func)
            loss, grads_and_vars = grad_function(self.ArchitectureObject, input_data, input_labels)
        print("-----grads and vars shape----")
        print(np.shape(grads_and_vars))
    return TowerResult(output, grads_and_vars, loss, diagnostics, input_labels, input_data)

  def _summarize_towers(self, tower_grads, losses, diagnostics):
    print("....")
    grads = self._average_gradients(tower_grads)
    print("....")
    diag, full_diag = self._average_diagnostics(diagnostics)
    print("....")
    #train_op = self._optimizer.apply_gradients(grads, global_step=self._global_step, name="ApplyGradientsTF")
    train_op = self._optimizer.apply_gradients(grads,name="ApplyGradients")
    print("....")
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    print("....")
    summary = tf.summary.merge(summaries)
    print("....")
    summed_losses = tf.reduce_sum(losses)
    return JoinedResult(summary, train_op, summed_losses, diag, full_diag)
