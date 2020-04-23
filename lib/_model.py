import tensorflow as tf
from .misc import printt
from contextlib import ExitStack
from .summaries import video_summary
import copy
import sys
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras import backend as K

class __Model__(object):
    def __init__(self, **kwargs):
        self.__tapes__ = None
        self.__losses__ = None # This is an active variable; (delete?) Issue #1.1
        self.__variables__ = None # This is an active variable; (delete?) Issue #1.1

        self.__optimisers__ = None # list
        self.__optimisers_models__ = None # list of dicts
        self.__forward_pass_model__ = None # Example: could be a concatentation
                                           # of __optimiser_models__

        class Config: pass
        self.__active_vars__ = Config()
        self.__active_vars__.summaries = False
        self.__active_vars__.verbose_summaries = False
        self.__active_vars__.training = False
        self.__active_vars__.validation = False
        self.__active_vars__.testing = False
        self.__active_vars__.return_weights = False
        self.__active_vars__.step = None
        self.__active_vars__.built = False # Only used in the case of not using GradientTape
                                           # which means that the Keras traininng API is
                                           # being used. Hence built=True indicates Keras
                                           # is being used. But please use
                                           # self.__gradient_taping__ instead.

        self.__analysis__ = Config()
        self.__analysis__.__active_vars__ = Config()


        self.__perception_config__ = Config()
        self.__perception_config__.debug = False
        self.__perception_config__.training = True
        #self.__perception_config__.printt = None
        self.__analysis_directory__ = None
        self.__kwargs__ = kwargs # Required for adding module arguments in JSON

        self.loss_func = self._loss_func if self.__perception_config__.debug is True else tf.function(self._loss_func)

    def __get_step__(self):
        return self.__active_vars__.step

    def module_arg(self, arg_name, instance_check=None, false_val=None, kwargs=None, force_user=False, convert_type=None):
        kwargs = self.__kwargs__ if kwargs is None else kwargs
        if convert_type is None:
          converter = lambda x : x
        else:
          converter = convert_type
        if force_user is True and arg_name not in kwargs.keys():
            printt("{} is required but not specified by the user".format(arg_name), error=True, stop=True)
        return false_val if not(
            arg_name in kwargs.keys() and (
                instance_check is None or (
                    isinstance(
                      kwargs[arg_name], instance_check
                    ) if convert_type is None else True)
                )) else converter(kwargs[arg_name])

    def add_summary(self, name, data, **kwargs):
        '''
        type: string from 'scalar', 'video', 'image'
        '''
        if 'type' not in kwargs.keys():
            printt("type not valid", error=True, stop=True)
        else:
            typ = kwargs['type']
        del(kwargs['type'])
        kwargs['step'] = self.__active_vars__.step
        #print("")
        #print("Some information about summaries")
        #print(self.__active_vars__)
        #print(self.__active_vars__.summaries)
        #print(self.__active_vars__.validation)
        #self.__active_vars__.summaries = tf.compat.v1.Print(self.__active_vars__.summaries, [self.__active_vars__.summaries])
        #print("")
        #tf.print(self.__active_vars__.summaries, output_stream=sys.stdout)
        '''
        Start
        '''
        '''
        cond_1 = tf.logical_and(
            self.__active_vars__.summaries,
            not('verbose' in kwargs.keys() and (kwargs['verbose'] == True))
        )
        cond_2 = tf.logical_and(
            self.__active_vars__.verbose_summaries == True,
            ('verbose' in kwargs.keys() and (kwargs['verbose'] == True))
        )
        cond_3 = ('force' in kwargs.keys()) and (kwargs['force'] == True)
        cond_final = tf.logical_or(tf.logical_or(cond_1, cond_2), cond_3)
        '''
        '''
        '''
        if not(
          ((self.__active_vars__.summaries == True) and (
            not('verbose' in kwargs.keys() and (kwargs['verbose'] == True)))
          ) or (
            ('verbose' in kwargs.keys() and (kwargs['verbose'] == True)) and (
              (self.__active_vars__.verbose_summaries == True))
          ) or (
            'force' in kwargs.keys() and (kwargs['force'] == True)
          )
        ):
            return
        else:
            if 'normalise' in kwargs.keys():
                if kwargs['normalise'] == True:
                    data = tf.math.real(data)
                    max_val = tf.reduce_max(data)
                    min_val = tf.reduce_min(data)
                    val_range = max_val - min_val
                    data = (data - min_val)/val_range
                    #data = data/max_val
                del(kwargs['normalise'])


            if self.__active_vars__.validation == True:
                name = "Validation/" + name
            else:
                if self.__active_vars__.verbose_summaries is True:
                    name = "Training_Verbose/" + name
                else:
                    name = "Training/" + name

            if 'verbose' in kwargs.keys():
                del(kwargs['verbose'])
            if 'force' in kwargs.keys():
                del(kwargs['force'])





            if typ == "scalar":
                return tf.summary.scalar(name, data, **kwargs)
            elif typ == "image":
                return tf.summary.image(name, data, **kwargs)
            elif typ == "video" or typ == "gif":
                return video_summary(name, data, **kwargs)
            else:
                printt("Invalid summary type", error=True)
                return


    def __forward_pass__(self, data, summaries=False, verbose_summaries=False, gradient_update=False, training=None):
        '''
        If `gradient_update' is True, the weights will be updated using the GradientTaping method.
        By default, it is assumed that this is being called by a training function, hence when training=None usually results in loss_func being called with argument training=True
        '''
        if gradient_update is False:
            if training is None:
                training = True
            all_trainable_variables, losses = self.loss_func(data,
                training=training, summaries=summaries, verbose_summaries=verbose_summaries,
                step=tf.convert_to_tensor(self.__active_vars__.step, dtype=tf.int64))
        elif gradient_update is True:
            n = len(self.__optimisers__)
            self.__tapes__ = [tf.GradientTape() for _ in range(n)]
            with ExitStack() as stack:
                for i, mgr in enumerate(self.__tapes__):
                    self.__tapes__[i] = stack.enter_context(mgr)
                all_trainable_variables, losses = self.loss_func(data,
                    training=True, summaries=summaries, verbose_summaries=verbose_summaries)
                self.__variables__ = all_trainable_variables # Issue #1.1
                self.__losses__ = losses # Issue #1.1

                for i, [tape, loss, optimiser_trainable_variables, optimiser] in enumerate(
                    zip(self.__tapes__,
                        self.__losses__,
                        self.__variables__,
                        self.__optimisers__)):
                    grads = tape.gradient(loss, optimiser_trainable_variables)
                    optimiser.apply_gradients(zip(grads, optimiser_trainable_variables))

    def __update_weights__(self, data, summaries=False, verbose_summaries=False, gradients=False):
        gradients = self.__gradient_taping__ if self.__gradient_taping__ is not None else gradients
        if gradients is True:
            '''
            Use Gradient Taping method
            '''
            self.__forward_pass__(data, summaries=summaries, verbose_summaries=verbose_summaries, gradient_update=True)
        else:
            '''
            Use Keras API
            '''
            if self.__active_vars__.built is False:
                '''
                Compile Models
                '''
                self.__optimisers_models__old__ = []
                self.__TEMP__trainfunctions = []
                for i, (optimizer, optimizer_models) in enumerate(zip(self.__optimisers__, self.__optimisers_models__)):
                    optimizer_models__models_old = optimizer_models["models"][:] # Copy in-list references
                    optimizer_models["models"] = [
                        self.__model_combiner__(
                            *optimizer_models["models"],
                            loss_function=optimizer_models["loss_function"]
                        )
                    ]
                    optimizer_models["models"][0].compile(
                        optimizer=optimizer,
                        loss=optimizer_models["keras_loss_functions"] if 'keras_loss_functions' in optimizer_models.keys() else None
                    )
                    if tf.__version__ == '2.2.0':
                        self.__TEMP__trainfunctions.append(optimizer_models["models"][0].make_train_function())
                    elif tf.__version__ == '2.1.0':
                        _,_,sampleweights_none = optimizer_models["models"][0]._standardize_user_data(
                            data, None, sample_weight=None, class_weight=None,
                            extract_tensors_from_dataset=True)
                        optimizer_models["models"][0]._update_sample_weight_modes(sample_weights=sampleweights_none) # is this needed?
                        optimizer_models["models"][0]._make_train_function()
                        self.__TEMP__trainfunctions.append(optimizer_models["models"][0].train_function)

                    optimizer_models_old = copy.copy(optimizer_models) # Shallow-copy of the dictionary's references
                    optimizer_models_old["models"] = optimizer_models__models_old
                    self.__optimisers_models__old__.append(optimizer_models_old)
                self.__active_vars__.built = True

            for i, optimizer_models in enumerate(self.__optimisers_models__):
                if tf.__version__ == '2.2.0' or tf.__version__ == '2.1.0':
                    if tf.__version__ == '2.1.0':
                        original_data = data
                        data,data_y_None,data_sampleweights_None = optimizer_models["models"][0]._standardize_user_data(
                            data, None, sample_weight=None, class_weight=None,
                            extract_tensors_from_dataset=True)
                        data = training_utils.ModelInputs(data).as_list()
                        data = data + list(data_y_None or []) + list(data_sampleweights_None or [])
                        if not isinstance(K.symbolic_learning_phase(), int):
                            data += [True]  # Add learning phase value.

                    vals = self.__TEMP__trainfunctions[i](data)
                    data = original_data
                else:
                    vals = optimizer_models["models"][0].train_on_batch(data)
                if (summaries or verbose_summaries):
                    self.__forward_pass__(data, summaries=summaries, verbose_summaries=verbose_summaries)

                metrics = {key: val for key,val in zip(optimizer_models["models"][0].metrics_names, vals)}
                return metrics


    class __model_combiner__(tf.keras.Model):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.models = args
            self.loss_function = kwargs['loss_function']

        def call(self, data, training=True, pass_number=None):
            for model in self.models:
                data = model(data)
            loss = self.loss_function(data)
            self.add_loss(loss)
            return data

    '''
    Mild modification perhaps neccessary
    '''

    def _loss_func(self, data, training=False, return_weights=False, validation=False,
        summaries=False, verbose_summaries=False, **kwargs):
        '''
        This function makes the appropriate __call__ of all required models,
        before then passing through the loss functions specified.

        Training: There is an option to run the models in training mode or not.
        The default behaviour of perception is to use the Keras static-graph-
        based training API so this loss_func is only used for logging training
        and validation summaries which are only forward passes (since summaries
        do not work with the Keras API). Hence, the default option is False.
        '''
        testing = True if (training is False and validation is False) else False

        '''
        Remember to run summary functions here
        '''
        self.__active_vars__.summaries = summaries
        self.__active_vars__.verbose_summaries = verbose_summaries
        self.__active_vars__.training = training
        self.__active_vars__.validation = validation
        self.__active_vars__.testing = testing
        self.__active_vars__.return_weights = return_weights
        if not(self.__gradient_taping__ is True):
            # Using Keras API:
            self.__active_vars__.step = kwargs['step']

        __optimisers_models__ = self.__optimisers_models__ if self.__gradient_taping__ is True else self.__optimisers_models__old__

        all_trainable_variables = []
        losses = []
        optimisers_diagnostics = []
        for i, models in enumerate(__optimisers_models__):
            '''
            Pass number below isn't a solid concept. But for example,
            pass_number = 0 could represent training a generator-discriminator
            whereas pass_number = 1 could represent training just the
            discriminator
            '''
            this_optimisers_diagnostics = []
            this_optimiser_trainable_variables = []
            results = data
            for model in models["models"]:
                results = model(results, training=training, pass_number=i)
                this_optimiser_trainable_variables += model.trainable_variables
                this_optimisers_diagnostics.append(results)

            all_trainable_variables.append(this_optimiser_trainable_variables)

            #print("Just after")
            #print(self.__active_vars__.validation)
            #print(self.__active_vars__.summaries)
            #print(self.__active_vars__)
            #print("")
            loss = models['loss_function'](results, pass_number=i)
            losses.append(loss)
            optimisers_diagnostics.append(this_optimisers_diagnostics)

        if training is True:
            return all_trainable_variables, losses
        else:
            if testing is True:
                return optimisers_diagnostics, losses
