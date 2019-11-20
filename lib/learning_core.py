import tensorflow as tf
import numpy as np
from architecture import architecture_base
import collections
TowerResult = collections.namedtuple('TowerResult', ('grads', 'loss', 'diagnostics'))

class learning_core(object):
    def __init__(self):
        self.ArchitectureObject = None

    def strap_architecture(self, ArchitectureObject):
        assert isinstance(Architectureobject, architecture) == True
        self.ArchitectureObject = ArchitectureObject

    def initialise_training(self):
        self._optimizer = self.ArchitectureObject.config.training.optimizer

    def single_tower(self, tower_ind, data, validation_graph=False):
        device_name_prefix = 'cpu' if self.cpu_only is True else 'gpu'

        with tf.device('/' + device_name_prefix + ':%d' % tower_ind):
            loss, diagnostics = self.ArchitectureObject.loss_func(data["extra_data"], validation_graph=validation_graph)
            grads_and_vars  = self._optimizer.compute_gradients(loss)
            printt("-----grads and vars shape----")
            printt(np.shape(grads_and_vars))
        return TowerResult(grads_and_vars, loss, diagnostics)

    def average_diagnostics(self, diagnostics_all_towers):
        '''
        diagnostics_all_towers is a list indexed by the GPU in which the
        (duplicated) models execute
        '''

        nGPUs = len(diagnostics_all_towers)
        keys = list(diagnostics_all_towers[0].keys())

        diagnostics = {}
        full_diagnostics = {}

        for key in keys:
            vars = [diagnostics_all_towers[i][key] for i in range(nGPUs)]
            full_diagnostics[key] = tf.convert_to_tensor(vals)
            '''
            Shape of this key in the dictionary is now:
            [GPU, <shape of diagnostic>]
            '''

            if('min' in key):
                diagnostics[key] = tf.reduce_min(input_tensor=vals)
            elif('max' in key):
                diagnostics[key] = tf.reduce_max(input_tensor=vals)
            else:
                diagnostics[key] = tf.reduce_mean(input_tensor=vals)

            this_shape = full_diagnostics[key].get_shape().as_list()
            printt(">>>>>>>>> Averaging Diagostics")
            printt(this_shape)

            '''
            [CHECK]
            NEEDS LOOKING AT:
            If the statistic contains multiple dimensions, then we need to
            ensure that

            if(len(this_shape)>1):
                nGPUs = this_shape[0]
                del(this_shape[0])
                this_shape[0] = this_shape[1] * nGPUs
                full_diagnostics[key] = tf.reshape(full_diagnostics[key], this_shape )
            '''
            this_shape = full_diagnostics[key].get_shape().as_list()
            printt(this_shape)
        return diagnostics, full_diagnostics



  def average_gradients(self, tower_grads):
    """Calculate the average gradient for each variable across all towers.

    Args:
      tower_grads: List of gradient lists for each tower. Each gradient list
        is a list of (gradient, variable) tuples for all variables.
    Returns:
      List of pairs of (gradient, variable) where the gradient has been
      averaged across all towers.
    """
    average_grads = []

    '''
    Loop through each GPU and print Variables and Gradients
    '''
    for grads_and_vars in tower_grads:
        this_shape = np.shape(grads_and_vars)
        for x, y in grads_and_vars:
            if(x!=None and hasattr(x, 'name')):
                printt('x' + x.name)

            if(y!=None and hasattr(y, 'name')):
                printt('variable name: ' + y.name)

    n_vars = 0 # Number of Variables in the model
    '''
    Loop through each variable and associated gradients, retrieve values for
    all GPUs
    '''
    for grads_and_vars in zip(*tower_grads):
      nGPUs = grads_and_vars.shape[0]

      grads = [variable_gradient for variable_gradient, variable_value in grads_and_vars if variable_gradient != None]
      grads = tf.stack(grads) # (along axis=0)
      grad = tf.reduce_mean(input_tensor=grads, axis=0)

      '''
      Get variable of GPU 0. Element 0 = Grad. Element 1 = Var.
      '''
      #value = grads_and_vars[0][1] # [CHECK] []
      values = [grads_and_vars[i] for i in range(nGPUs)]
      value = tf.reduce_mean(values, axis=0)

      n_vars += np.prod(v.get_shape().as_list())

      printt("====== %s" % v.name)
      printt(grad.get_shape().as_list())
      printt(value.get_shape().as_list(), end=" = ")
      printt(np.prod(value.get_shape().as_list()))
      grad_and_var = (grad, value)
      average_grads.append(grad_and_var)
    print("THERE ARE %d WEIGHTS/BIASES IN THIS MODEL" % n_vars)
    return average_grads
