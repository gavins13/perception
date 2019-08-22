import tensorflow as tf

class learning_core(object):
  def __init__(self):
    self.ArchitectureObject = None

  def strap_architecture(self, ArchitectureObject):
    self.ArchitectureObject = ArchitectureObject
    hparams = ArchitectureObject.hparam

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
    for grads_and_vars in zip(*tower_grads):
      grads = tf.stack([g for g, _ in grads_and_vars])
      grad = tf.reduce_mean(grads, 0)

      v = grads_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
    return average_grads



  def evaluate_classification(model_output, gt_output, this_tower_scope, num_targets=1, loss_type='margin'):
    # [] - would be good to rewrite this in the future [unfinished]
    # [] to delete num_targets? - num targets is the number of classifications made per label
  with tf.name_scope('loss'):
    if loss_type == 'margin':
      classification_loss = _margin_loss(gt_output, model_output)
    else:
      raise NotImplementedError('Not implemented')

    with tf.name_scope('total'):
      batch_classification_loss = tf.reduce_mean(classification_loss, axis=0)
      tf.add_to_collection('losses', batch_classification_loss)
  tf.summary.scalar('batch_classification_cost', batch_classification_loss)

  all_losses = tf.get_collection('losses', this_tower_scope) # list of tensors returned
  total_loss = tf.add_n(all_losses, name='total_loss') # element-wise addition of the list of tensors
  tf.summary.scalar('total_loss', total_loss)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      _, targets = tf.nn.top_k(gt_output, k=num_targets)
      _, predictions = tf.nn.top_k(model_output, k=num_targets)
      missed_targets = tf.sets.set_difference(targets, predictions) # i.e. return only the values which are different (not the same as minus!)
      num_missed_targets = tf.sets.set_size(missed_targets)
      correct = tf.equal(num_missed_targets, 0)
      almost_correct = tf.less(num_missed_targets, num_targets)
      correct_sum = tf.reduce_sum(tf.cast(correct, tf.float32))
      almost_correct_sum = tf.reduce_sum(tf.cast(almost_correct, tf.float32))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
  tf.summary.scalar('accuracy', accuracy)
  tf.summary.scalar('correct_prediction_batch', correct_sum)
  tf.summary.scalar('almost_correct_batch', almost_correct_sum)
  return total_loss, correct_sum, almost_correct_sum
