import tensorflow as tf
from .misc import printt
from contextlib import ExitStack
from .summaries import video_summary
import copy
import sys
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras import backend as K
import time
import os
import inspect
from .customiser import CustomUserModule

class __Model__(CustomUserModule):
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
                                           # If built=True, the model has compiled,
                                           # train functions have been made (soon to also
                                           # include test, validation functions) and
                                           # a forward pass through the model has been made
        self.__active_vars__.saving_intialised = False

        self.__analysis__ = Config()
        self.__analysis__.__active_vars__ = Config()


        self.__perception_config__ = Config()
        self.__perception_config__.debug = False
        self.__perception_config__.training = True
        self.__perception_config__.reset_optimisers = False
        #self.__perception_config__.printt = None
        self.__analysis_directory__ = None
        self.__kwargs__ = kwargs # Required for adding module arguments in JSON

        self.loss_func = self._loss_func if self.__perception_config__.debug is True else tf.function(self._loss_func)

    def __get_step__(self):
        return self.__active_vars__.step

    def __finalise__(self):
        '''
        Code to run before initialisation of the Keras models during first pass
        '''
        for i, models in enumerate(self.__optimisers_models__):
            models["__validation_flag__"] = [False]*len(models["models"])
            models["__training_flag__"] = [False]*len(models["models"])
            for j, model in enumerate(models["models"]):
                args, varargs, varkw, defaults = inspect.getargspec(model.call)
                val_flag = True if 'validation' in args else False
                train_flag = True if 'training' in args else False
                print(i,j,"training flag, validation flag", train_flag, val_flag)
                models["__validation_flag__"][j] = val_flag
                models["__training_flag__"][j] = train_flag
        pass

    def add_summary(self, name, data, **kwargs):
        #tf.summary.trace_off()
        '''
        type: string from 'scalar', 'video', 'image'
        '''
        if 'type' not in kwargs.keys():
            printt("type not valid", error=True, stop=True)
        else:
            typ = kwargs['type']
        del(kwargs['type'])
        kwargs['step'] = self.__active_vars__.step

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




            with tf.summary.experimental.summary_scope("", default_name="") as (tag, scope):
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

    def __build__(self, *args, **kwargs):
        printt("Building model", info=True)
        kwargs = {**kwargs, _no_training_updates: True}
        _ = self.__update_weights__(self, *args, **kwargs)
        return

    def __reset_optimisers__(self):
        printt("Optimisers are being reset!", debug=True)
        self.__optimisers_before_reset__= self.__optimisers__[:]
        for optimiser in self.__optimisers__:
            for var in optimiser.variables():
                var.assign(tf.zeros_like(var))
        '''
        self.__optimisers_before_reset__= self.__optimisers__[:]
        for i, optimiser in enumerate(self.__optimisers__):
            config = optimiser.get_config()
            cls = optimiser.__class__
            self.__optimisers__[i] = None
            optimiser = cls.from_config(config)
        '''

    def __built__(self):
        return self.__active_vars__.built

    def __build__(self):
        self.__active_vars__.built = False
        self.__update_weights__(None, _no_training_updates=True)

    def __build_once__(self):
        self.__update_weights__(None, _no_training_updates=True)


    def __update_weights__(self, data, summaries=False, verbose_summaries=False, gradients=False, _no_training_updates=False):
        gradients = self.__gradient_taping__ if self.__gradient_taping__ is not None else gradients
        if gradients is True:
            '''
            Use Gradient Taping method
            '''
            if self.__perception_config__.reset_optimisers is True:
                print("Optimisers have been asked to reset but method not implemented in Perception 2.0 yet for Gradient Taped training", stop=True, error=True)
            self.__forward_pass__(data, summaries=summaries, verbose_summaries=verbose_summaries, gradient_update=True)
        else:
            '''
            Use Keras API
            '''
            if self.__active_vars__.built is False:
                '''
                Reset optimisers if requested
                '''
                if self.__perception_config__.reset_optimisers is True:
                    self.__reset_optimisers__()
                    
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
                            loss_function=optimizer_models["loss_function"],
                            validation_flags=optimizer_models["__validation_flag__"],
                            training_flags=optimizer_models["__training_flag__"]
                        )
                    ]

                    optimizer_models["models"][0].compile(
                        optimizer=optimizer,
                        loss=optimizer_models["keras_loss_functions"] if 'keras_loss_functions' in optimizer_models.keys() else None
                    )

                    '''
                    if self.__perception_config__.reset_optimisers is True:
                        self.__reset_optimisers__()
                        optimizer_models["models"][0].compile(
                            optimizer=optimizer,
                            loss=optimizer_models["keras_loss_functions"] if 'keras_loss_functions' in optimizer_models.keys() else None
                        )
                    '''

                    optimizer_models["__validation_flag__"] = [True]
                    optimizer_models["__training_flag__"] = [True]
                    if tf.__version__[0:5] == '2.2.0' or tf.__version__[0:3] == '2.3':
                        self.__TEMP__trainfunctions.append(optimizer_models["models"][0].make_train_function())
                    elif tf.__version__[0:5] == '2.1.0':
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

            if _no_training_updates == True:
                return {}
            for i, optimizer_models in enumerate(self.__optimisers_models__):
                if tf.__version__[0:5] == '2.2.0' or tf.__version__[0:5] == '2.1.0' or tf.__version__[0:3] == '2.3':
                    original_data = data
                    if tf.__version__[0:5] == '2.1.0':
                        data,data_y_None,data_sampleweights_None = optimizer_models["models"][0]._standardize_user_data(
                            data, None, sample_weight=None, class_weight=None,
                            extract_tensors_from_dataset=True)
                        data = training_utils.ModelInputs(data).as_list()
                        data = data + list(data_y_None or []) + list(data_sampleweights_None or [])
                        if not isinstance(K.symbolic_learning_phase(), int):
                            data += [True]  # Add learning phase value.
                    elif tf.__version__[0:5] == '2.2.0' or tf.__version__[0:3] == '2.3':
                        data = tf.compat.v1.data.make_one_shot_iterator(tf.data.Dataset.from_tensors(data))
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
            self.validation_flags = kwargs['validation_flags']
            self.training_flags = kwargs['training_flags']

        def call(self, data, training=False, pass_number=None, validation=False):
            for model, val_flag, train_flag in zip(self.models, self.validation_flags, self.training_flags):
                dict_ = {}
                if train_flag is True:
                    dict_["training"] = training
                if val_flag is True:
                    dict_["validation"] = validation
                data = model(data, **dict_)
            loss = self.loss_function(data)
            self.add_loss(loss, inputs=True)
            return data


    def save(self, path, data=None, initialised=True):
        printt("Saving Temporarily disabled.", info=True)
        return
        '''
        TEMPORARILY DISABLED

        This function will save the TensorFlow models so that they can be
        reloaded for deployment/simple access purposes.

        In the case where Keras is being used, if the active variable `built`
        is `True` then the model is ready for saving and active variable
        `saving_initialised` can be set to True. If not, then a forward pass
        needs to be made using `data` through the compiled Keras model in
        `__optimiser_models__` using the __forward_pass__ method with the
        argument `training` set to `False` (although it should not matter since
        the conditioning introduced by the training argument creates conditional
        branches which are also saved). Remember to also loop through the
        old models in `__optimisers_models__old__`.

        In the case where the GradientTaping method is used, raise a
        NotImplementedError().
        '''
        start_time = time.time()
        for i, models in enumerate(self.__optimisers_models__):
            print("Saving Optimised Model {}".format(i))
            if self.__active_vars__.saving_intialised is False:
                #self.__forward_pass__(data, training=False)
                if self.__active_vars__.built is True:
                    self.__active_vars__.saving_intialised = True
                else:
                    if data is None:
                        printt("Model needs building. Please provide data.", stop=True, error=True)
                    self.__build__(data)
                    #printt("Saving not implemented for gradient taping method", error=True, stop=True)
            file_format = '.tf'
            for j, model in enumerate(models["models"]):
                print("Saving combined submodel {}".format(j))
                this_filename = str(i) + "_" + str(j) + "_" + str(model.name)
                this_filename = os.path.join(path, this_filename + '' + file_format)
                model.predict(data)
                model.save(this_filename)

            for j, model in enumerate(self.__optimisers_models__old__[i]["models"]):
                print("Saving submodel {}".format(j))
                this_filename = "_" + str(i) + "_" + str(j) + "_" + str(model.name)
                this_filename = os.path.join(path, this_filename + '' + file_format)
                model.save(this_filename)
        duration = time.time() - start_time
        printt("Model Saved. Time taken: {:.3f}".format(duration), info=True)
        return

    '''
    Mild modification perhaps neccessary
    '''

    def _loss_func(self, data, training=False, return_weights=False, validation=False,
        summaries=False, verbose_summaries=False, step=None, **kwargs):
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
            self.__active_vars__.step = step #kwargs['step']

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
            for model, training_flag, validation_flag in zip(models["models"], models["__training_flag__"], models["__validation_flag__"]):
                dict_ = {}
                if training_flag is True:
                    dict_["training"] = training
                if validation_flag is True:
                    dict_["validation"] = validation
                results = model(results, pass_number=i, **dict_)
                this_optimiser_trainable_variables += model.trainable_variables
                this_optimisers_diagnostics.append(results)
            all_trainable_variables.append(this_optimiser_trainable_variables)

            loss = models['loss_function'](results, pass_number=i)
            losses.append(loss)
            optimisers_diagnostics.append(this_optimisers_diagnostics)

        if training is True:
            return all_trainable_variables, losses
        else:
            if testing is True:
                return optimisers_diagnostics, losses

    @classmethod
    def from_tf_model(cls, model,
        epochs=300,
        checkpoint_steps=100,
        validation_steps=100,
        summary_steps=10,
        verbose_summary_steps=50,
        learning_rate=1.e-3,
        optimizer=tf.keras.optimizers.Adam
    ):
        '''
        Use this method to automate the creation of a perception model from
        a TensorFlow or Keras-based model.
        '''
        assert isinstance(model, tf.keras.Model)
        PerceptionModel = cls()
        PerceptionModel.__config__.checkpoint_steps = checkpoint_steps
        PerceptionModel.__config__.validation_steps = validation_steps
        PerceptionModel.__config__.summary_steps = summary_steps
        PerceptionModel.__config__.verbose_summary_steps = verbose_summary_steps
        PerceptionModel.__config__.epochs = epochs
        PerceptionModel.__config__.test_epochs = 1
        PerceptionModel.__config__.saved_model_epochs = 1
        PerceptionModel.__config__.print_training_metrics = False

        __keras_model__ = model

        PerceptionModel.__forward_pass_model__ = __keras_model__
        PerceptionModel.__optimisers__ = [optimizer(learning_rate)]
        PerceptionModel.__optimisers_models__ = [
            {
                'models': [__keras_model__],
                'loss_function': lambda *args, **kwargs : 0.
            }
        ]
        
    @classmethod
    def from_keras_model(cls, *args, **kwargs):
        return cls.from_tf_model(*args, **kwargs)