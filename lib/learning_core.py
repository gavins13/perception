import tensorflow as tf
from losses import margin_loss as main_loss_function
import numpy as np


class learning_core(object):
  def __init__(self):
    self.ArchitectureObject = None

  def strap_architecture(self, ArchitectureObject):
    self.ArchitectureObject = ArchitectureObject
    hparams = ArchitectureObject.hparams

    with tf.device('/cpu:0'):
      self._global_step = tf.get_variable(
          'global_step', [],
          initializer=tf.constant_initializer(0),
          trainable=False)

      learning_rate = tf.train.exponential_decay(
          learning_rate=hparams.learning_rate,
          global_step=self._global_step,
          decay_steps=hparams.decay_steps,
          decay_rate=hparams.decay_rate)
      learning_rate = tf.maximum(learning_rate, 1e-6)

      self._optimizer = tf.train.AdamOptimizer(learning_rate)

  def _average_gradients(self, tower_grads):
    """Calculate the average gradient for each variable across all towers.

    Args:
      tower_grads: List of gradient lists for each tower. Each gradient list
        is a list of (gradient, variable) tuples for all variables.
    Returns:
      List of pairs of (gradient, variable) where the gradient has been
      averaged across all towers.
    """
    average_grads = []
    def replace_none_with_zero(l):
        '''If None: There is no connection from input to output.
        There is a connection, but it's through a discrete variable with meaningless gradients.
        There is a connection, but it's through an op that doesn't have an implemented gradient.
        '''
        pass
    for grads_and_vars in tower_grads: # for each tower
        this_shape = np.shape(grads_and_vars)
        for x, y in grads_and_vars:
            if(x!=None and hasattr(x, 'name')):
                print('x' + x.name)

            if(y!=None and hasattr(y, 'name')):
                print('variable name: ' + y.name)
    #stop()
    # if you use `in tower_grads` then it will loop over each Tower
    # but if you use `in zip(*tower_grads)`, it will loop over the variables
    for grads_and_vars in zip(*tower_grads): # for each tower
      gs = [variable_gradient for variable_gradient, variable_value in grads_and_vars if variable_gradient != None]
      grads = tf.stack(gs)# for each pair in the e.g. (32, 2) list
      # [] [unfinished] [check if it right to use gradients only not equal to None]

      grad = tf.reduce_mean(grads, 0)
      v = grads_and_vars[0][1] # i.e. the variable value # i.e Tower 0, element 1=variable   (becaus element 0 = gradient)
      print("====== %s" % v.name)
      print(grad.get_shape().as_list())
      print(v.get_shape().as_list())
      grad_and_var = (grad, v) # (average_grad_across_all_tower, the variable value)
      average_grads.append(grad_and_var)
    return average_grads


  def loss(self,x,y,margin=0.4, downweight=0.5):
      labels= x
      raw_logits = y
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


  def evaluate_classification(self,model_output, gt_output, this_tower_scope, num_targets=1, loss_type='margin'):
    # [] - would be good to rewrite this in the future [unfinished]
    # [] to delete num_targets? - num targets is the number of classifications made per label
    print(">>>> Find Margin Loss")
    with tf.name_scope('loss'):
        if loss_type == 'margin':
          classification_loss = main_loss_function(gt_output, model_output)
        else:
          raise NotImplementedError('Not implemented')

    print(">>>> Find Mean Loss")
    with tf.name_scope('total'):
      print(">>>>>> Batch Loss")
      print(classification_loss.get_shape().as_list())
      batch_classification_loss = tf.reduce_mean(classification_loss)
      print(">>>>>> Add to collection")
      tf.add_to_collection('losses', batch_classification_loss)
      print(">>>>>> Creating summary")
      #batch_classification_loss=tf.constant(batch_classification_loss)
      print(batch_classification_loss.get_shape().as_list())
      print(batch_classification_loss)
      tmp=tf.contrib.summary.scalar(name='batch_classification_cost', tensor=batch_classification_loss)

    print(">>>> Add result to collection of loss results for this tower")
    all_losses = tf.get_collection('losses') # [] , this_tower_scope) # list of tensors returned
    total_loss = tf.add_n(all_losses, name='total_loss') # element-wise addition of the list of tensors
    #print(total_loss.get_shape().as_list())
    tf.contrib.summary.scalar('total_loss', total_loss)

    print(">>>> Find Accuracy")
    #%model_output = tf.cast(model_output, np.float32)
    #gt_output = tf.cast(gt_output, np.float32)
    print(np.shape(gt_output))
    print(model_output.get_shape().as_list())
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            print(">>>>> Find GT classifications")
            #tmp, targets = tf.nn.top_k(gt_output, num_targets)
            #tmp, targets = tf.nn.top_k(gt_output, num_targets)
            #targets = tf.argmax(gt_output, axis=1)
            targets = tf.argmax(gt_output, axis=1)
            print(">>>>> Find Model classifications")
            #tmp, predictions = tf.nn.top_k(model_output, num_targets)
            predictions = tf.argmax(model_output, axis=1)
            print(">>>>> Find Difference in classifications")
            #targets = tf.transpose(targets, [1,0])
            #predictions = tf.transpose(predictions, [1,0])
            targets = tf.expand_dims(targets, 1)
            predictions = tf.expand_dims(predictions, 1)
            print(model_output.dtype)
            print(gt_output.dtype)
            targets = tf.cast(targets, tf.float32)
            predictions = tf.cast(predictions,  tf.float32)
            print(targets.dtype)
            print(predictions.dtype)
            print(targets.get_shape().as_list())
            print(predictions.get_shape().as_list())
            #tmp=tf.subtract(gt_output, model_output)
            print(">>>>>>> Set Difference")
            #missed_targets = tf.sets.set_difference(gt_output, model_output) # i.e. return only the values which are different (not the same as minus!)
            #missed_targets = tf.contrib.metrics.set_difference([[1],[1],[1],[4],[5]],[[2],[1],[1],[3],[5]]) # i.e. return only the values which are different (not the same as minus!)
            #missed_targets = tf.contrib.metrics.set_difference(targets, predictions) # i.e. return only the values which are different (not the same as minus!)
            #gt_output_indx = tf.constant(gt_output_indx)
            #print(gt_output_indx)
            #print(np.shape(gt_output_indx))
            #tmp=tf.subtract([[1,1],[3,4],[5,4]],[[2,7],[7,2],[8,5]])
            tmp=tf.subtract(targets, predictions)
            #tmp = tf.subtract([[1],[1],[1],[4],[5]],[[2],[1],[1],[3],[5]])
            '''tmp = tf.nn.in_top_k(model_output, gt_output_indx, num_targets)'''
            print(">>>>>>> Set Size")
            #num_missed_targets = tf.sets.set_size(missed_targets)
            tmp=tf.abs(tmp)
            print(".")
            tmp=tf.cast(tf.greater(tmp, 0), tf.float32)
            print(".")
            print(tmp.get_shape().as_list())
            num_missed_targets = tf.reduce_sum(tmp)
            print(">>>>>>> Set Equal")
            correct = tf.equal(num_missed_targets, 0)
            print(">>>>> Find Correct/Incorrect")
            almost_correct = tf.less(num_missed_targets, num_targets)
            correct_sum = tf.reduce_sum(tf.cast(correct, tf.float32))
            almost_correct_sum = tf.reduce_sum(tf.cast(almost_correct, tf.float32))
    print(">>>> Add results to output")
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.contrib.summary.scalar('accuracy', accuracy)
        tf.contrib.summary.scalar('correct_prediction_batch', correct_sum)
        tf.contrib.summary.scalar('almost_correct_batch', almost_correct_sum)
    print(">>>> return results")
    return total_loss, correct_sum, almost_correct_sum
