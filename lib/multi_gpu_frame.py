import tensorflow as tf
import numpy as np


import sys
sys.path.insert(0, 'lib/')

from learning_core import learning_core

import collections
TowerResult = collections.namedtuple('TowerResult', ('result', 'almost',
                                                     'correct', 'grads'))


JoinedResult = collections.namedtuple('JoinedResult', ('summary', 'train_op',
                                                       'correct', 'almost'))



class multi_gpu_model(learning_core):
  def __init__(self, ArchitectureObject=None, cpu_only=False):
    self.ArchitectureObject = None
    self.cpu_only = cpu_only
    if(ArchitectureObject is not None):
      self.strap_architecture(ArchitectureObject)
    else:
      print(" ... Time to load architecture ... ")



  def run_multi_gpu(self, DataObject, num_gpus=1):
    """Build the Graph and add the train ops on multiple GPUs.

    Divides the inference and gradient computation on multiple gpus.
    Then aggregates the gradients and return the resultant ops.

    Args:
      features: A list of dictionary of different features of input data.
                len(features) should be at least num_gpus.
      num_gpus: Number of gpus to be distributed on.
    Returns:
      A tuple of JoinedResult output Ops to be called in Session.run for
      training, evaluation or visualization, such as train_op and merged
      summary and a list of inferred outputs of each tower.
    """

    DataObject.set_num_gpus(num_gpus)
    print(">>>>Using %d GPUs" % num_gpus)

    if(self.ArchitectureObject is None):
        raise Exception('problem with architecture: not loaded')
    almosts = []
    corrects = []
    tower_grads = []
    results = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(num_gpus):
        print('>>Assignment of data to tower/GPU %d' % i)
        input_data, input_labels = DataObject.get_data(gpu=i)
        print('>>>Data Shapes')
        print(np.shape(input_data))
        print(np.shape(input_labels))
        tower_output = self._single_tower(i, input_data, input_labels)
        results.append(tower_output.result)
        almosts.append(tower_output.almost)
        corrects.append(tower_output.correct)
        tower_grads.append(tower_output.grads)

    print('>> Sumarise results from all towers')
    summarized_results = self._summarize_towers(almosts, corrects, tower_grads)
    print('>> Return results from all towers')
    return summarized_results, results

  def _single_tower(self, tower_ind, input_data, input_labels, num_targets=1):
    """Calculates the model gradient for one tower.

    Adds the inference and loss operations to the graph. Calculates the
    gradients based on the loss. Appends all the output values of this tower to
    their respective lists.

    Args:
      tower_ind: The index number for this tower. Each tower is named as
                  tower_{tower_ind} and resides on gpu:{tower_ind}.
      feature: Dictionary of batched features like images and labels.
    Returns:
      A namedtuple TowerResult containing the inferred values like logits and
      reconstructions, gradients and evaluation metrics.
    """
    if(self.ArchitectureObject is None):
        raise Exception('problem with architecture: not loaded')

    if(self.cpu_only==True):
        device_name_prefix = 'cpu'
    else:
        device_name_prefix = 'gpu'

    with tf.device('/' + device_name_prefix + ':%d' % tower_ind):
      with tf.name_scope('tower_%d' % (tower_ind)) as scope:
        print(">>>Building Architecture for tower %d..." % tower_ind)
        res = self.ArchitectureObject.build(input_data)
        print(">>>Finished Building Architecture.")
        print(">>>Calculate classification results... (eval)")
        losses, correct, almost = self.evaluate_classification(
            model_output=res.output,
            gt_output=input_labels,
            num_targets=num_targets,
            this_tower_scope=scope,
            loss_type=self.ArchitectureObject.hparams.loss_type)
        print(">>>Finsished Calculating classification results.")
        tf.get_variable_scope().reuse_variables()
        grads = self._optimizer.compute_gradients(losses)
    return TowerResult(res, almost, correct, grads)

  def _summarize_towers(self, almosts, corrects, tower_grads):
      # []  need to go over and rewrite
    """Aggregates the results and gradients over all towers.

    Args:
      almosts: The number of almost correct samples for each tower.
      corrects: The number of correct samples for each tower.
      tower_grads: The gradient list for each tower.

    Returns:
      A JoinedResult of evaluation results, the train op and the summary op.
    """

    grads = self._average_gradients(tower_grads)
    train_op = self._optimizer.apply_gradients(
        grads, global_step=self._global_step)
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    summary = tf.summary.merge(summaries)
    stacked_corrects = tf.stack(corrects)
    stacked_almosts = tf.stack(almosts)
    summed_corrects = tf.reduce_sum(stacked_corrects, 0)
    summed_almosts = tf.reduce_sum(stacked_almosts, 0)
    return JoinedResult(summary, train_op, summed_corrects, summed_almosts)
