'''
class Execution:
    - load Config.perception file
    - load Experiments.perception file
    - Checkpointing, restoration, writing the saves to files
    - Keeps track of experiments by writing to a experiments txt file
    - Loads tensorboard during execution using port number specified in the exp config using no GPUs and
'''
from .dataset import Dataset
from .model import Model
import json
import os

with open(os.path.join(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../"),
    "Config.perception"
  ), "r") as config_file:
    Config = json.load(config_file)
from .misc import printt, Logger

from contextlib import ExitStack, nullcontext
import sys
from datetime import datetime
from .model import Model
import tensorflow as tf
import time
from tensorboard import program as tb_program
import numpy as np
import inspect
from tabulate import tabulate
import csv


from .experiments import Experiments

class Execution(object):
    def __init__(self, *args, **kwargs):
        '''
        Arguments:
        dataset: Dataset object (optional)
        experiment_name: string (optional)
        save_folder: string (optional)
        model: Model object
        tensorboard_only: bool (optional)
        '''
        printt(kwargs, debug=True)

        '''
        Training or Testing
        '''
        if 'experiment_type' in kwargs.keys():
            exp_type = kwargs['experiment_type']
            if exp_type == 'train':
                self.__call__func = self.training
            elif exp_type == 'testing' or exp_type == 'evaluate':
                self.__call__func = self.testing
            else:
                printt("Invalid Experiment Execution type set", error=True, stop=True)
        else:
            printt("Experiment Execution type not set", error=True, stop=True)
        if 'tensorboard_only' in kwargs.keys() and kwargs['tensorboard_only'] is True:
            self.__call__func = self.tensorboard_only
            kwargs['execute'] = True

        '''
        Tensorboard port
        '''
        self.tensorboard_port = None
        if 'tensorboard_port' in kwargs.keys() and kwargs['tensorboard_port'] is not None:
            self.tensorboard_port = kwargs['tensorboard_port']

        '''
        Handle Dataset
        '''
        printt("Loading dataset...", info=True)
        if 'dataset' in kwargs.keys():
            # Use this dataset
            if isinstance(kwargs['dataset'], Dataset) is False:
                printt("Instance check for chosen Dataset failed.", warning=True)
            self.Dataset = kwargs['dataset']
        else:
            self.Dataset = Dataset()
            self.Dataset.use('developer_mode', 'cifar10')
            self.Dataset.create()
        printt("Finished loading dataset...", info=True)
        
        '''
        Handle Model
        '''
        if 'model' in kwargs.keys() and (issubclass(kwargs['model'], Model) or\
          inspect.isclass(kwargs['model'])):
            model_args = {}
            if 'model_args' in kwargs.keys() and kwargs['model_args'] is not None:
                model_args = kwargs['model_args']
            if 'debug' in kwargs.keys() and kwargs['debug'] in [True, False]:
                debug = kwargs['debug']
            kwargs['model'] = kwargs['model'](training=exp_type, **model_args)
            kwargs['model'].__perception_config__.training = exp_type
            kwargs['model'].__perception_config__.debug = debug
        if 'model' in kwargs.keys():
            # Use this dataset
            if isinstance(kwargs['model'], Model) is False:
                printt("Instance check for chosen Model failed.", warning=True)
            self.Model = kwargs['model']
            self.Model.create_models()
            self.Model.__finalise__()
        else:
            printt(kwargs.keys(), debug=True)
            printt(kwargs['model'], debug=True)
            printt("No Model for execution", error=True, stop=True)
        step_counter = tf.Variable(initial_value=1, trainable=False,
                                   name="global_step", dtype=tf.int64)
        current_dataset_file_counter = tf.Variable(initial_value=0, trainable=False,
                                   name="current_file_counter", dtype=tf.int64)
        epoch_counter = tf.Variable(initial_value=0, trainable=False,
                                   name="epoch_counter", dtype=tf.int64)

        '''
        Handle directories for saving
        '''
        # Create Perception Experiments Save Directory
        perception_save_path = Config['save_directory'] if not((
            'perception_save_path' in kwargs.keys() )
             and kwargs['perception_save_path'] is not None
             ) else kwargs['perception_save_path']
        if not(os.path.exists(perception_save_path)):
            os.makedirs(perception_save_path)
        # Create an experiment name that is unique
        self.experiment_name = kwargs['experiment_name'] \
            if 'experiment_name' in kwargs.keys() and \
            isinstance(kwargs['experiment_name'], str) else 'tmp'
        printt("Experiment Name: {0}".format(self.experiment_name), info=True)
        datetimestr = str(datetime.now())
        datetimestr = datetimestr.replace(" ", "-")

        # Saving section
        save_folder = None if not(('save_folder' in kwargs.keys()) and\
            (isinstance(kwargs['save_folder'], str) is True)) else kwargs['save_folder']
        printt("Execution: Save folder: {}".format(save_folder), info=True)
        # Check for reset flag
        if ('reset' in kwargs.keys()) and (kwargs['reset'] is True):
            save_folder = None
            if 'experiments_manager' in kwargs.keys():
                ExperimentsManager = kwargs['experiments_manager']
            else:
                ExperimentsManager = Experiments()
        # Create the save folder name from the experiment name above
        if(save_folder is None):
            self.save_directory_name = self.experiment_name + '_' + datetimestr
            self.save_directory = os.path.join(perception_save_path,
                self.save_directory_name)
            printt("New save folder created: {}".format(self.save_directory_name), info=True)
        else:
            self.save_directory_name = save_folder
            self.save_directory = os.path.join(perception_save_path, save_folder)
            printt("Load Dir being used: {}".format(self.save_directory_name), info=True)

        # Create the save folder
        if not(os.path.exists(self.save_directory)):
            os.makedirs(self.save_directory)
            if 'experiments_manager' in kwargs.keys():
                ExperimentsManager = kwargs['experiments_manager']
            else:
                ExperimentsManager = Experiments()

        # If reset, or new folder created for experiment, update the exp.
        # If custom perception_save_path used, update this as well
        self.experiment_id = kwargs['experiment_id']
        if ('experiment_id' in kwargs.keys()) and ('ExperimentsManager' in locals()):
            ExperimentsManager[kwargs['experiment_id']] = self.save_directory_name
            if ((
             'perception_save_path' in kwargs.keys() )\
             and (kwargs['perception_save_path'] is not None)
             ):
                ExperimentsManager.update_experiment(
                 kwargs['experiment_id'], 'perception_save_path',
                 kwargs['perception_save_path'])

        '''
        [] [check]
        This block of code for the experiments manager needs to be reviewed.
        '''
        if 'ExperimentsManager' not in locals():
            if 'experiments_manager' in kwargs.keys():
                ExperimentsManager = kwargs['experiments_manager']
            else:
                ExperimentsManager = Experiments()
        self.ExperimentsManager = ExperimentsManager

        if (exp_type == 'train') and \
          hasattr(ExperimentsManager[self.experiment_id], 'training_finished') and \
          (ExperimentsManager[self.experiment_id]["training_finished"] == True):
            printt("Experiment has already finished training", stop=True, error=True)

        
        if (exp_type == 'evaluate') and \
          hasattr(ExperimentsManager[self.experiment_id], 'evaluation_finished') and \
          (ExperimentsManager[self.experiment_id]["evaluation_finished"] == True) and \
          not(
              ("ignore_json_evaluation_finished" in kwargs.keys()) and \
              (kwargs["ignore_json_evaluation_finished"] is True)
          ):
            printt("Experiment has already finished evaluation", stop=True, error=True)

        '''
        Create summary writer
        '''
        # Note: the save directory is also created here with `makedirs`
        if not(os.path.exists(os.path.join(self.save_directory, 'summaries'))):
            os.makedirs(os.path.join(self.save_directory, 'summaries'))
        self.summary_writer = tf.summary.create_file_writer(
            os.path.join(self.save_directory, 'summaries')) if (exp_type == 'train') else tf.summary.create_noop_writer()
        self.summaries_directory = os.path.join(self.save_directory, 'summaries')

        '''
        Sort out Perception logging
        '''
        printt_experiment_log = os.path.join(self.save_directory, 'log.txt')
        #self.print = lambda *args, **kwargs : printt(*args, **{**kwargs, 'full_file_path': printt_experiment_log if 'full_file_path' not in kwargs.keys() else kwargs['full_file_path']})
        #globals()['printt'] = self.print
        #self.Model.print = self.print
        sys.stdout = Logger(printt_experiment_log)
        self.metrics_enabled = False if not('metrics_enabled' in kwargs.keys()) else kwargs['metrics_enabled']
        self.metrics_printing_enabled = True if not('metrics_printing_enabled' in kwargs.keys()) else kwargs['metrics_printing_enabled']
        self.print_model_png = False if not('print_model_png' in kwargs.keys()) else kwargs['print_model_png'] # [] should be True

        '''
        Create checkpointer and restore
        '''
        if not(os.path.exists(os.path.join(self.save_directory, 'checkpoints'))):
            os.makedirs(os.path.join(self.save_directory, 'checkpoints'))
        checkpoint_dir = os.path.join(self.save_directory, 'checkpoints')
        optimisers = {'opt_'+str(i): opt for i, opt in enumerate(
            self.Model.__optimisers__)}
        models = {'model_'+str(i): model for i, model in enumerate(
            self.Model.__keras_models__)}

        self.ckpt = tf.train.Checkpoint(**optimisers, **models, step=step_counter, current_file=current_dataset_file_counter, epoch=epoch_counter)
        if (('load_weights_folder_name' in kwargs.keys()) and (
          kwargs['load_weights_folder_name'] is not None
        )) and (save_folder is None):
            initialise_weights_save_directory = os.path.join(perception_save_path, kwargs['load_weights_folder_name'])
            initialise_weights_checkpoint_dir = os.path.join(initialise_weights_save_directory, 'checkpoints')
            #self.initialise_weights_ckpt_manager = tf.train.CheckpointManager(
            #    self.ckpt, initialise_weights_checkpoint_dir, max_to_keep=3)
            #status = self.ckpt.restore(self.initialise_weights_ckpt_manager.latest_checkpoint)
            #self.ckpt_manager = tf.train.CheckpointManager(
            #    self.ckpt, checkpoint_dir, max_to_keep=3)
            self.ckpt_manager = tf.train.CheckpointManager(
                self.ckpt, checkpoint_dir, max_to_keep=3)
            load_weights_file = tf.train.latest_checkpoint(initialise_weights_checkpoint_dir)
            status = self.ckpt.restore(load_weights_file)
            if load_weights_file is not None:
                printt("Restored Weights from a separate experiment from {}".format(load_weights_file),
                    info=True)
                #status.assert_consumed()
                #status.assert_existing_objects_matched()
            else:
                printt("Loading weights from another experiment failed.", error=True, stop=True)
        else:
            self.ckpt_manager = tf.train.CheckpointManager(
                self.ckpt, checkpoint_dir, max_to_keep=3)
            status = self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint is not None:
            printt("Restored from {}".format(self.ckpt_manager.latest_checkpoint),
                info=True)
            #status.assert_consumed()
            #status.assert_existing_objects_matched()
        else:
            printt("Initializing from scratch.", debug=True)
        self.Model.__active_vars__.step = step_counter

        '''
        Reset Optimisers in Perception Model?
        '''
        self.Model.__perception_config__.reset_optimisers = (
            ('reset_optimisers' in kwargs.keys()) and\
            (kwargs['reset_optimisers'] is True)) and (save_folder is None)


        '''
        Create Saved Model and directory
        '''
        if not(os.path.exists(os.path.join(self.save_directory, 'saved_model'))):
            os.makedirs(os.path.join(self.save_directory, 'saved_model'))
        self.saved_model_directory = os.path.join(self.save_directory, 'saved_model')

        '''
        Create analysis directory
        '''
        if not(os.path.exists(os.path.join(self.save_directory, 'analysis'))):
            os.makedirs(os.path.join(self.save_directory, 'analysis'))
        self.analysis_directory = os.path.join(self.save_directory, 'analysis')
        self.Model.__analysis_directory__ = self.analysis_directory

        '''
        Some model information
        '''
        self.gradient_taping = ('gradient_taping' in kwargs.keys()) and\
            (kwargs['gradient_taping'] is True)

        '''
        Running validation on CPU?
        '''
        self.validation_on_cpu = (('validation_on_cpu' in kwargs.keys()) and (
            kwargs['validation_on_cpu'] is True))
        if self.validation_on_cpu is False:
            printt("Validation is operating on the GPU. For some applications, this will invoke the building of a new graph on the GPU and thus occupy more memory leading to training issues. Please use the 'validation_on_cpu' option in Experiment() or use command-line flag", warning=True)
        '''
        Saving the model only?
        '''
        if 'save_only' in kwargs.keys() and kwargs['save_only'] is True:
            printt("Start saving...", info=True)
            self.__call__func = self.save_only

        '''
        Start training?
        '''
        self.tb_url = None
        if 'execute' in kwargs.keys() and kwargs['execute'] is True:
            printt("Start training/testing", info=True)
            self()
    def __call__(self, *args, **kwargs):
        return self.__call__func(*args, **kwargs)

    def testing(self):
        n_epochs = 1
        if hasattr(self.Model.__config__, 'test_epochs'):
            n_epochs = self.Model.__config__.test_epochs
        printt("Entering testing loop", debug=True)
        epochs = 0
        step = 0
        built = False
        while epochs < n_epochs:
            for record_number, data_record in self.Dataset.test_dataset.enumerate():
                start_time = time.time()
                if built == False:
                    self.Model.__update_weights__(data_record, _no_training_updates=True)
                    built = True

                #diagnostics = self.Model.loss_func(data, training=False)
                #analysis_output = self.Model.analyse(diagnostics, step, self.analysis_directory)

                diagnostics, analysis_output = self.run(data_record,
                    return_analysis=True,
                    return_diagnostics=True, step=step)
                '''
                Print to console and save checkpoint
                '''
                data_split_num = record_number if self.Dataset.system_type.use_generator is False else self.Dataset.current.test_file
                data_split_num_total = self.Dataset.test_dataset_steps if self.Dataset.system_type.use_generator is False else self.Dataset.generator.num_test_files
                print("testing epoch: %d" % epochs, end=";")
                print("data split: %d of %d" % (data_split_num, data_split_num_total), end=";")
                print("step: %d of %d" % (step+1, self.Dataset.test_dataset_steps), end=";")
                duration = time.time() - start_time
                print("time: {}".format(duration))
                print("")
                sys.stdout.write("\033[K")
                step += 1
            epochs += 1
        self.Model.analysis_complete()
        self.ExperimentsManager.update_experiment(
            self.experiment_id, 'evaluation_finished',
            True)

    def run(self, input_data, return_diagnostics=True, execute_analysis=False,
        return_analysis=False, step=None, analysis_directory=None):
        '''
        Use this to run forward passes on data
        Arguments:
         - input_data : list of input data to be executed
         -
        '''
        self.Model.__build_once__()

        analysis_directory = self.analysis_directory if analysis_directory is None else analysis_directory

        execute_analysis = True if return_analysis is True else execute_analysis

        diagnostics, losses = self.Model.loss_func(input_data, training=False)
        if execute_analysis is True:
            analysis_output = self.Model.analyse(diagnostics, step, analysis_directory)

        if return_diagnostics is True:
            if return_analysis is True:
                return diagnostics, analysis_output
            else:
                return diagnostics
        else:
            if return_analysis is True:
                return analysis_output

    def tensorboard(self, port=None):
        '''
        Start tensorboard
        '''
        if port is None:
            port = self.tensorboard_port
        tb = tb_program.TensorBoard()
        args = [None, '--logdir', self.summaries_directory, '--host', '0.0.0.0']
        if port is not None:
            args.append('--port')
            args.append(str(port))
        tb.configure(argv=args)
        tb_url = tb.launch()
        print("TensorBoard started at {}".format(tb_url))
        self.tb_url = tb_url
    def tensorboard_only(self):
        if self.tb_url is None:
            self.tensorboard()
        input("Press enter to exit and stop TensorBoard...")

    def logging(self, val):
        full_file_path = os.path.join(self.save_directory, "model_summary.txt")
        #printt("Printing model summary to path: {}".format(full_file_path), debug=True)
        printt(val, full_file_path=full_file_path)

    def training(self):
        '''
        Training Loop
        '''
        printt("Entering training loop", debug=True)
        epochs = 0
        step=0
        #epochs = int(tf.floor(tf.divide(self.ckpt.step, self.Dataset.train_dataset_steps)))
        epochs = int(self.ckpt.epoch)
        step=int(self.ckpt.step) # Starts on 1
        train = True
        #self.Dataset.train_dataset=self.Dataset.train_dataset.skip(step)
        self.Dataset.skip(step, current_file=int(self.ckpt.current_file), epoch=int(self.ckpt.epoch))

        if (epochs >= self.Model.__config__.epochs) and (self.Model.__config__.epochs != -1):
            train = False
        '''
        Metrics (and printing) enabled?
        '''
        if self.metrics_enabled is not True:
            tf.keras.Model.add_metric = lambda *args, **kwargs : None
            tf.keras.layers.Layer.add_metric = lambda *args, **kwargs : None
        else:
            metrics_filename = os.path.join(self.save_directory, "metrics.csv")
            metrics_file_ = open(metrics_filename, 'a')
            metrics_file = csv.writer(metrics_file_)
        '''
        Start Tensorboard
        '''
        self.tensorboard()

        print("Current File: {} \nEpoch: {} \nStep: {} \n".format(int(self.ckpt.current_file), int(self.ckpt.epoch), step))
        with self.summary_writer.as_default():
            with tf.summary.record_if(True):
                while train is True: # Trains across epochs, NOT steps
                    for record_number, data_record in\
                        self.Dataset.train_dataset.enumerate(): # Avoid tqdm progress bar
                            add_summary = (step+1) % self.Model.__config__.summary_steps
                            add_summary = True if add_summary == 0 else False
                            verbose_add_summary = (step+1) % self.Model.__config__.verbose_summary_steps
                            verbose_add_summary = True if verbose_add_summary == 0 else False
                            # Start Timing
                            start_time = time.time()
                            # Execute Model
                            metrics = self.Model.__update_weights__(data_record, summaries=add_summary, verbose_summaries=verbose_add_summary, gradients=self.gradient_taping)
                            # Print metrics if enabled
                            if self.metrics_enabled is True:
                                if self.metrics_printing_enabled is True:
                                    print("")
                                    #print(tabulate([metrics.values()], headers=metrics.keys()))
                                    print(tabulate(metrics.items(), headers=['Metric', 'Value']))
                                if step == 1:
                                    metrics_file.writerow(metrics.keys())
                                metrics_file.writerow(list(metrics.values()))
                                metrics_steps = int(self.Model.__config__.summary_steps / 10.)+1 # Just so that we get metrics more often
                                add_metrics_ = (step+1) % metrics_steps
                                if add_summary is True or add_metrics_ is True:
                                    metrics_file_.flush()
                            # Print training information and duration
                            print("training epoch: {}".format(epochs+1), end=";")
                            data_split_num = record_number if self.Dataset.system_type.use_generator is False else self.Dataset.current.file
                            data_split_num_total = self.Dataset.train_dataset_steps if self.Dataset.system_type.use_generator is False else self.Dataset.generator.num_files
                            print("data split: %d of %d" % (
                                data_split_num+1,
                                data_split_num_total), end=";")
                            print("step: %d / %d" % (step,self.Dataset.train_dataset_steps), end=";")


                            # Validation
                            if step % self.Model.__config__.validation_steps == 0:
                                this_step = tf.convert_to_tensor(self.Model.__active_vars__.step, dtype=tf.int64)
                                for validation_data_record in self.Dataset.validation_dataset: #.take(self.Dataset.validation_dataset_length):
                                    with tf.device('/cpu:0') if self.validation_on_cpu is True else nullcontext():
                                        self.Model.loss_func(validation_data_record, training=False,
                                            validation=True, summaries=True, verbose_summaries=True,
                                            step=this_step
                                        )
                                        print("validation")
                                duration = time.time() - start_time
                                print("Validation time: %.3f" % duration)

                            # Stop Timing
                            duration = time.time() - start_time
                            print("time: %.3f" % duration, end=";")


                            # Checkpointing
                            if step %\
                             self.Model.__config__.checkpoint_steps == 0:
                                self.ckpt.step.assign_add(self.Model.__config__.checkpoint_steps)
                                if self.Dataset.system_type.use_generator is True:
                                    self.ckpt.current_file.assign(self.Dataset.current.file)
                                self.ckpt.epoch.assign(self.Dataset.current.epoch)
                                save_path = self.ckpt_manager.save()
                                print("Saved checkpoint for step {}".format(
                                    int(self.ckpt.step)), end=";")
                                sys.stdout.write("\033[K")
                            print("", end="\r")




                            # Save summary of the model
                            if (step == 1): #or ((step+1) == self.Model.__config__.summary_steps):
                                self.Model.__forward_pass_model__.summary(print_fn=self.logging)
                                for optimisers_models in self.Model.__optimisers_models__:
                                    #for model in optimisers_models['models']:
                                        #model.summary(print_fn=self.logging)
                                    _get_summary(optimisers_models['models'], self.logging, self.save_directory if self.print_model_png is True else None)

                            # Increment to next step
                            step += 1
                            self.Model.__active_vars__.step = step

                            if self.Dataset.system_type.use_generator is False:
                                self.Dataset.current.epoch = epochs
                                self.Dataset.current.step = step
                            elif self.Dataset.system_type.use_generator is True:
                                epochs = self.Dataset.current.epoch
                                if (epochs >= self.Model.__config__.epochs) and (self.Model.__config__.epochs != -1):
                                    train = False
                                    break # Exit For loop iterating over generator dataset

                    epochs += 1
                    if epochs % self.Model.__config__.saved_model_epochs == 0:
                        # NOTE: There is a bug caused by the generator dataset being in an endless
                        #       for loop which will cause this next bit of code not to run. Lines
                        #       involve "train=False" have been copied to a special generator mode
                        #       case above (lines 536-540)
                        this_epoch_saved_model_dir = os.path.join(self.saved_model_directory, 'epoch_'+str(epochs))
                        if not(os.path.exists(this_epoch_saved_model_dir)):
                            os.makedirs(this_epoch_saved_model_dir)
                        saving_enabled = False if (tf.__version__ == '2.0.0') else True
                        saving_enabled = False if (self.gradient_taping == True) else saving_enabled
                        #saving_enabled = False # [CHECK] []
                        if saving_enabled is True:
                            self.Model.save(this_epoch_saved_model_dir)
                            print("TensorBoard started at {}".format(self.tb_url))
                    if (epochs >= self.Model.__config__.epochs) and (self.Model.__config__.epochs != -1):
                        train = False
        self.ExperimentsManager.update_experiment(
         self.experiment_id, 'training_finished',
         True)
        tf.keras.backend.clear_session()
        self.tensorboard_only()

    def save_only(self):
        '''
        Load epoch number
        '''
        #epochs = 0
        #step=0
        epochs = int(self.ckpt.epoch)
        step=int(self.ckpt.step) # Starts on 1
        self.Dataset.skip(step, current_file=int(self.ckpt.current_file), epoch=int(self.ckpt.epoch))
        print("Current File: {} \nEpoch: {} \nStep: {} \n".format(int(self.ckpt.current_file), int(self.ckpt.epoch), step))
        '''
        Run one training iteration, without weight updates
        '''
        with self.summary_writer.as_default():
            for record_number, data_record in\
                self.Dataset.train_dataset.take(1).enumerate():
                    # Execute Model
                    metrics = self.Model.__update_weights__(data_record, gradients=self.gradient_taping, _no_training_updates=True)
        '''
        Save the model at the epoch number loaded
        '''
        this_epoch_saved_model_dir = os.path.join(self.saved_model_directory, 'epoch_'+str(epochs))
        if not(os.path.exists(this_epoch_saved_model_dir)):
            os.makedirs(this_epoch_saved_model_dir)
        saving_enabled = False if (tf.__version__ == '2.0.0') else True
        saving_enabled = False if (self.gradient_taping == True) else saving_enabled
        #saving_enabled = False # [CHECK] []
        if saving_enabled is True:
            self.Model.save(this_epoch_saved_model_dir, data=data_record)
        else:
            printt("TF must be > 2.0 and GradientTaping must be off", stop=True, error=True)
        print("Model saved to: {}".format(this_epoch_saved_model_dir))
        return

def _get_summary(models, logging, dir, i=0):
    for model in models:
        if isinstance(model, tf.keras.Model) is True:
            model.summary(print_fn=logging)
            if dir is not None:
                tf.keras.utils.plot_model(model, to_file=os.path.join(dir,"model_"+str(i)+".png"), show_shapes=True, expand_nested=True)
            i=i+1
            if hasattr(model, 'models') is True and (isinstance(model.models, list) or isinstance(model.models, tuple)):
                _get_summary(model.models, logging, dir, i=i)
        elif (isinstance(model, list) or isinstance(model, tuple)):
            _get_summary(model, logging, dir, i=i)