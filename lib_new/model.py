import tensorflow as tf

'''
Model

Model allows use to define complex models with potentially multiple training
objectives without needing to worry about so much boilerplate code. However,
the boilerplate is designed to be customisable for cases where the framework
provided here isn't flexible enough.

To get started, inherit Model in your own work.

Step 1:
Define create_models() which tells the TF which models to use, which optimisers
to use and how the optimisers should be applied to the models. Usually, only
a single optimiser and model will be required. However, there are cases such as
with GANs where the generator and discriminator will need to be designed as two
separate models. This is because whilst the generator-discrimator can be
trained singularly, the discriminator needs to also be trained separately to
discriminate the real images. Hence, two optimisers are needed. One operates on
the two models (the generator + discriminator) whereas the second optimiser
operates only on the discriminator. Also, state which loss function to use - 
Note: the loss function receives the output of the last model call.

Step 2:
Define Models (e.g. CustomModel) just as with Keras and define loss functions
(e.g. loss_function).

Step 3:
Define self.__config__ variables (mainly regarding the number of saving steps
for validation, summaries, etc...)

Step 4:
Define analyse(diagnostics, idx, save_dir)
'''

class Model(object):
	'''
	__init__: sets up the optimizers, learning rate, hyperparameters
	'''
	def __init__(self):
		self.__tapes__ = None
		self.__losses__ = None # This is an active variable; (delete?) Issue #1.1
		self.__variables__ = None # This is an active variable; (delete?) Issue #1.1
		self.__optimisers__ = None

		class Config: pass
		self.__config__ = Config()
		self.__config__.checkpoint_steps = 100
		self.__config__.validation_steps = 20
		self.__config__.summary_steps = 20
		self.__config__.epochs = 300
		self.__config__.saved_model_epochs = 1

		self.__keras_models__ = None

		self.__active_vars__ = Config()
		self.__active_vars__.summaries = False
		self.__active_vars__.training = False
		self.__active_vars__.validation = False
		self.__active_vars__.return_weights = False
	class CustomModel(tf.keras.Model):
		def call(self, inputs, training=False, pass_number=None):
			pass

	def create_models(self):
		'''
		Create all keras models required and append them self.__kereas_models__
		See example below
		This is called during the Execution intialisation
		'''
		self.__keras_models__ = []
		self.__keras_models__.append(self.CustomModel())
		self.__optimisers__ = []
		self.__optimisers__.append(tf.keras.optimizers.Adam(5.e-5))

		'''
		This specifies the models which the optimisers will operate on
		'''
		self.__optimiser_models__ = []
		self.__optimiser_models__.append({
			'models': self.__keras_models___,
			'loss_function': self.loss_function
			})
		pass

	def loss_function(self, results, pass_number=None):
		'''
		Literally just return loss
		'''
		pass

	def analyse(self, diagnostics, idx, save_dir):
        '''
        Save all diagnostics to a pickle
        '''
        with open(save_dir + 'individual_pkle/' + str(idx) + '.p', 'wb') as handle:
            pickle.dump(diagnostics, handle)
            
        pass
		'''
        Create sub directories for each type of analysis
        '''
        ground_truth_path = os.path.join(save_dir, 'ground_truth')
        if not(os.path.exists(ground_truth_path)):
            os.makedirs(ground_truth_path)

    '''
    Mild modification perhaps neccessary
    '''
	def loss_func(self, data, training=False, return_weights=False, validation=False,
		summaries=False):
		'''
		Remember to run summary functions here
		'''
		self.__active_vars__.summaries = summaries
		self.__active_vars__.training = training
		self.__active_vars__.validation = validation
		self.__active_vars__.return_weights = return_weights

		all_trainable_variables = []
		losses = []

		for i, models in enumerate(self.__optimiser_models__):
			'''
			Pass number below isn't a solid concept. But for example,
			pass_number = 0 could represent training a generator-discriminator
			whereas pass_number = 1 could represent training just the
			discriminator
			'''
			results = data
			for model in models.models:
				results = model(results, training=training, pass_number=i)
			loss = models['loss_function'](results, pass_number=i)
			all_trainable_variables.append(model.trainable_variables)
			losses.append(loss)

		testing = True if (training is False and validation is False)
		if training is True:
			return all_trainable_variables, losses
		else:
			if testing is True:
				return diagnostics
	'''
	No modification required
	'''

	def add_summary(self, name, data, type='scalar'):
		'''
		type: string from 'scalar', 'video', 'image'
		'''
		if self.__active_vars__.summaries == False:
			return
		else:
			raise NotImplemented()
	

	def __update_weights__(self, data):
		n = len(self.__optimisers__)
		self.__tapes__ =[tf.Gradient]*n
        with ExitStack() as stack:
            for i, mgr in enumerate(self.tapes):
                self.tapes[i] = stack.enter_context(mgr)
        	all_trainable_variables, losses = self.loss_func(data, training=True)
    		self.__variables__ = all_trainable_variables # Issue #1.1
    		self.__losses__ = losses # Issue #1.1

    		for i, [tape, loss, all_trainable_variables, optimiser] in enumerate(
    			zip(self.__tapes__,
    				self.__losses__,
    				self.__variables__,
    				self.__optimisers__)):
    			grads = tape.gradient(loss, variables)
				optimiser.apply_gradients(zip(grads, variables))

