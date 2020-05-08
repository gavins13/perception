import tensorflow as tf
from .misc import printt
from ._model import __Model__
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
(e.g. loss_function). When adding summaries with the loss_function, be sure to
use add_summary() function

Step 3:
Define self.__config__ variables (mainly regarding the number of saving steps
for validation, summaries, etc...)

Step 4:
Define analyse(diagnostics, idx, save_dir)

NOTE: within your tf.keras.Model class, if you are going to intialise another
tf.keras.Model, then please append the end result to tf.keras.Model attribute
called self.models. This will allow accurate .summary()s to be produced.
'''

class Model(__Model__):
    '''
    This is an abstract interface class
    __init__: sets up the hyperparameters, summary frequency, etc...
    '''
    def __init__(self, **kwargs):
        '''
        Always start by calling super() initialisation and for logging,
        also declare `globals()['print'] = self.print`
        '''
        super().__init__(**kwargs)
        self.__gradient_taping__ = None # If None or False, assume Keras API used. If None, it will use the `gradients` argument passed to __update_weights__

        class Config: pass
        self.__config__ = Config()
        self.__config__.checkpoint_steps = 100
        self.__config__.validation_steps = 20
        self.__config__.summary_steps = 20
        self.__config__.verbose_summary_steps = 40
        self.__config__.epochs = 300
        self.__config__.test_epochs = 1
        self.__config__.saved_model_epochs = 1
        self.__config__.print_training_metrics = False

    class CustomModel(tf.keras.Model):
        '''
        Add decorator below if you want to use SavedModel and TF Serving
        '''
        @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
        def call(self, inputs, training=False, pass_number=None):
            pass

    def create_models(self):
        '''
        Create all keras models required and append them to the dictionary
        `__optimiser_models__`
        This is called during the Execution intialisation
        '''
        __keras_model__ = self.CustomModel()
        self.__forward_pass_model__ = __keras_model__
        self.__optimisers__ = []
        self.__optimisers__.append(tf.keras.optimizers.Adam(5.e-5))

        '''
        This specifies the models which the optimisers will operate on
        '''
        self.__optimiser_models__ = []
        self.__optimiser_models__.append({
            'models': __keras_model___,
            'loss_function': self.loss_function
            })
        pass

    def loss_function(self, results, pass_number=None):
        '''
        Literally just return single scalar loss. Please log summaries in this
        function using the perception self.add_summary() method. Do not use
        TensorFlow summaries directly without understanding the consequences.
        Note: do not invoke within Keras models if using the Keras API (i.e.
        perception gradient_taping is False and debug is False).

        If there are parts to the loss_function that are useful for the testing
        then, please call the loss_function in the forward_pass_model which
        can be defined in a custom way.

        Note: since debug is False by default, the loss_func will run as a
        graph.

        If parts used to calculate the loss
        are useful for the analyse() function, then please store in the
        active variable self.__analysis__.__active_vars__ and remember,
        that there is access to the variable self.__active_vars__.testing
        to tell the model is the model is in testing mode or not
        '''
        pass


    def analyse(self, diagnostics, idx, save_dir):
        '''
        Save all diagnostics to a pickle.
        Note: diagnostics has the same nested structure as
        self.__optimiser_models__ containing the network outputs for each

        If there are parts to the loss_function that are useful for the testing
        then, please call the loss_function in the forward_pass_model which
        can be defined in a custom way. I.e. include an `analysis` argument
        which allows the function to return not only a scalar but instead a
        list of useful Tensors
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

    def analysis_complete(self):
        '''
        This runs at the end of testing. Usually it involves working with a
        variable(s) produced during testing in order to produce some value
        (e.g. mean, std, etc...) and probably will make use of the variable:
        self.__analysis__.__active_vars__
        '''
        pass
