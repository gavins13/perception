import tensorflow as tf
import numpy as np


import sys
sys.path.insert(0, 'lib/')

from learning_core import learning_core

import collections
TowerResult = collections.namedtuple('TowerResult', ('result', 'almost',
                                                     'correct', 'grads', 'loss'))


JoinedResult = collections.namedtuple('JoinedResult', ('summary', 'train_op',
                                                       'correct', 'almost', 'total_loss'))


class Namespace: pass


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
    losses = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(num_gpus):
        print('>>Assignment of data to tower/GPU %d' % i)
        input_data, input_labels = DataObject.get_data(gpu=i)
        print('>>>Data Shapes')
        print(np.shape(input_data))
        print(np.shape(input_labels))
        tower_output = self._single_tower(i, input_data, input_labels)
        print(">>>Grad shapes")
        results.append(tower_output.result)
        almosts.append(tower_output.almost)
        corrects.append(tower_output.correct)
        tower_grads.append(tower_output.grads)
        losses.append(tower_output.loss)

    print('>> Sumarise results from all towers')
    summarized_results = self._summarize_towers(almosts, corrects, tower_grads, losses)
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
        #with tf.GradientTape() as tape:
        tf.get_variable_scope().reuse_variables()
        #main_res, main_almost, main_correct = [],[] , []
        ns = Namespace()
        def losses_func(the_model_object=self.ArchitectureObject, input_images=input_data, the_target=input_labels,  margin=0.4, downweight=0.5):

            print(">>>Start Building Architecture.")
            res = the_model_object.build(input_images)
            print(">>>Finished Building Architecture.")



            print(">>>Calculate classification results... (eval)")
            losses, correct, almost = self.evaluate_classification(
                model_output=res.output,
                gt_output=the_target,
                num_targets=num_targets,
                this_tower_scope=scope,
                loss_type=the_model_object.hparams.loss_type)
            ns.res = res
            ns.almost = almost
            ns.correct = correct
            print(">>>Finsished Calculating classification results.")

            labels= the_target
            raw_logits = res.output
            print(">>>>> Subtract 0.5 from all classifications")
            print(raw_logits.get_shape().as_list())
            print(np.shape(labels))
            logits = raw_logits - 0.5
            print(">>>>> Find Positive Cost")
            positive_cost = labels * tf.cast(tf.less(logits, margin),
                                             tf.float32) * tf.pow(logits - margin, 2)
            print(">>>>> Find Negative Cost")
            negative_cost = (1 - labels) * tf.cast(
                tf.greater(logits, -margin), tf.float32) * tf.pow(logits + margin, 2)
            print(">>>>> Return loss")
            class_loss= tf.add(tf.multiply( 0.5, positive_cost),  tf.multiply(tf.multiply(downweight,0.5), negative_cost))
            batch_loss = tf.reduce_mean(class_loss)

            return batch_loss
        print(">>> Use AdamOptimizer built in gradient finder")
        if(self.eager==False):
          with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            loss = losses_func()
            grads_and_vars  = self._optimizer.compute_gradients(loss) # [] [unfinished] why
        else:
            #grads_and_vars  = self._optimizer.compute_gradients(losses_func)
            #print(">>> Use contrib.eager gradient finder")
            grad_function = tf.contrib.eager.implicit_value_and_gradients(losses_func)
            loss, grads_and_vars = grad_function(self.ArchitectureObject, input_data, input_labels)
        print("-----grads and vars shape----")
        print(np.shape(grads_and_vars))
        '''grads = grads_and_vars
        print(np.shape(grads))
        print(np.shape(grads[0]))
        print(grads[0])
        print(np.shape(grads[1]))
        print(grads[1][5])'''
    return TowerResult(ns.res, ns.almost, ns.correct, grads_and_vars, loss)

  def _summarize_towers(self, almosts, corrects, tower_grads, losses):
      # []  need to go over and rewrite
    """Aggregates the results and gradients over all towers.

    Args:
      almosts: The number of almost correct samples for each tower.
      corrects: The number of correct samples for each tower.
      tower_grads: The gradient list for each tower.

    Returns:
      A JoinedResult of evaluation results, the train op and the summary op.
    """
    print("....")
    grads = self._average_gradients(tower_grads)
    print("....")
    #train_op = self._optimizer.apply_gradients(grads, global_step=self._global_step, name="ApplyGradientsTF")
    train_op = self._optimizer.apply_gradients(grads,name="ApplyGradientsTF")
    print("....")
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    print("....")
    #summary = tf.contrib.summary.merge(summaries)
    summary = tf.contrib.summary.all_summary_ops()
    print("....")
    stacked_corrects = tf.stack(corrects)
    print("....")
    stacked_almosts = tf.stack(almosts)
    print("....")
    summed_corrects = tf.reduce_sum(stacked_corrects, 0)
    print("....")
    summed_almosts = tf.reduce_sum(stacked_almosts, 0)
    print("....")
    summed_losses = tf.reduce_sum(losses)
    return JoinedResult(summary, train_op, summed_corrects, summed_almosts, summed_losses)
