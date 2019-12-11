import tensorflow as tf

class Model(object):
	'''
	__init__: sets up the optimizers, learning rate, hyperparameters
	'''
	def __init__(self):
		self.__tapes__ = None
		self.__losses__ = None
		self.__variables__ = None
		self.__optimisers__ = None

	
	def loss_func(self, data, training=False, return_weights=False):
		self.build(training=training)
		if training is True:
			return all_variables, losses, optimisers
		else:
			return diagnostics


	def __update_weights__(self, data):
		self.__tapes__ =[tf.Gradient]*n
		def wrapper():
	        with ExitStack() as stack:
	            for i, mgr in enumerate(self.tapes):
	                self.tapes[i] = stack.enter_context(mgr)
            	all_variables, losses, optimisers = self.loss_func(data, training=True)


        		for i, [tape, loss, all_variables, optimiser] in enumerate(
        			zip(self.__tapes__,
        				self.__losses__,
        				self.__variables__,
        				self.__optimisers__)):
        			grads = tape.gradient(loss, variables)
    				optimiser.apply_gradients(zip(grads, variables))

            	grads = tape.gradient(loss, self.Model.trainable_weights)
                        optimizer.apply_gradients(zip(grads, Model.trainable_weights))
                        steps += 1
	        return wrapper