import tensorflow as tf
import numpy as np


class learning_core(object):
  def __init__(self):
    self.ArchitectureObject = None

  def strap_architecture(self, ArchitectureObject):
    self.ArchitectureObject = ArchitectureObject
    self.hparams = ArchitectureObject.hparams
  def initialise_training(self):
    with tf.device('/cpu:0'):
      self._global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False) # [] [unfinished]
      #self._global_step=np.array(0)
      learning_rate = tf.train.exponential_decay(
          learning_rate=self.hparams.learning_rate,
          global_step=self._global_step,
          decay_steps=self.hparams.decay_steps,
          decay_rate=self.hparams.decay_rate)
      learning_rate = tf.maximum(learning_rate, self.hparams.maximum_learning_rate)
      self.learning_rate = learning_rate

      if(hasattr(self.ArchitectureObject, 'AdamEpsilon')):
          AdamEpsilon = self.ArchitectureObject.AdamEpsilon
      else:
          AdamEpsilon = 1.e-8
      self._optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=AdamEpsilon)


  def _average_diagnostics(self, diagnostics_all_towers):
    n = len(diagnostics_all_towers)
    keys=list(diagnostics_all_towers[0].keys())
    diagnostics = {}
    full_diagnostics = {}
    for key in keys:
        vals = []
        for i in range(n):
            vals.append(diagnostics_all_towers[i][key])
        if('min' in key):
            diagnostics[key] = tf.reduce_min(vals)
        elif('max' in key):
            diagnostics[key] = tf.reduce_max(vals)
        else:
            diagnostics[key] = tf.reduce_mean(vals)
        full_diagnostics[key] = tf.convert_to_tensor(vals)
        #full_diagnostics[key] = vals
        this_shape = full_diagnostics[key].get_shape().as_list()
        print(">>>>>>>>> Averaging Diagostics")
        print(this_shape)
        if(len(this_shape)>1):
            this_shape_first = this_shape[0]
            del(this_shape[0])
            this_shape[0] = this_shape[0] * this_shape_first
            full_diagnostics[key] = tf.reshape(full_diagnostics[key], this_shape )
        this_shape = full_diagnostics[key].get_shape().as_list()
        print(this_shape)
    return diagnostics, full_diagnostics



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
    n_vars = 0
    for grads_and_vars in zip(*tower_grads): # for each tower
      gs = [variable_gradient for variable_gradient, variable_value in grads_and_vars if variable_gradient != None]
      grads = tf.stack(gs)# for each pair in the e.g. (32, 2) list
      # [] [unfinished] [check if it right to use gradients only not equal to None]

      grad = tf.reduce_mean(grads, 0)
      v = grads_and_vars[0][1] # i.e. the variable value # i.e Tower 0, element 1=variable   (becaus element 0 = gradient)
      print("====== %s" % v.name)
      print(grad.get_shape().as_list())
      print(v.get_shape().as_list(), end=" = ")
      print(np.prod(v.get_shape().as_list()))
      n_vars += np.prod(v.get_shape().as_list())
      grad_and_var = (grad, v) # (average_grad_across_all_tower, the variable value)
      average_grads.append(grad_and_var)
    print("THERE ARE %d WEIGHTS/BIASES IN THIS MODEL" % n_vars)
    return average_grads
