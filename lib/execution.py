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
from .misc import printt

from contextlib import ExitStack
import sys
from datetime import datetime
from .model import Model
import tensorflow as tf
import time
from tensorboard import program as tb_program
import numpy as np
import inspect

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
        Handle Dataset
        '''
        if 'dataset' in kwargs.keys():
            # Use this dataset
            if isinstance(kwargs['dataset'], Dataset) is False:
                printt("Instance check for chosen Dataset failed.", warning=True)
            self.Dataset = kwargs['dataset']
        else:
            self.Dataset = Dataset()
            self.Dataset.use('developer_mode', 'cifar10')
            self.Dataset.create()

        '''
        Handle Model
        '''
        if 'model' in kwargs.keys() and (issubclass(kwargs['model'], Model) or\
          inspect.isclass(kwargs['model'])):
            model_args = {}
            if 'model_args' in kwargs.keys() and kwargs['model_args'] is not None:
                model_args = kwargs['model_args']
            kwargs['model'] = kwargs['model'](training=exp_type, **model_args)
        if 'model' in kwargs.keys():
            # Use this dataset
            if isinstance(kwargs['model'], Model) is False:
                printt("Instance check for chosen Model failed.", warning=True)
            self.Model = kwargs['model']
            self.Model.create_models()
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
            (isinstance(kwargs['save_folder'], str))) else kwargs['save_folder']
        # Check for reset flag
        if 'reset' in kwargs.keys() and kwargs['reset'] is True:
            save_folder = None
            if 'experiments_manager' in kwargs.keys():
                ExperimentsManager = kwargs['experiments_manager']
            else:
                ExperimentsManager = Experiments()
        # Create the save folder name from the experiment name above
        if(save_folder==None):
            self.save_directory_name = self.experiment_name + '_' + datetimestr
            self.save_directory = os.path.join(perception_save_path,
                self.save_directory_name)
        else:
            self.save_directory_name = save_folder
            self.save_directory = os.path.join(perception_save_path, save_folder)
            printt("Load Dir being used.", info=True)

        # Create the save folder
        if not(os.path.exists(self.save_directory)):
            os.makedirs(self.save_directory)
            if 'experiments_manager' in kwargs.keys():
                ExperimentsManager = kwargs['experiments_manager']
            else:
                ExperimentsManager = Experiments()

        # If reset, or new folder created for experiment, update the exp.
        # If custom perception_save_path used, update this as well
        if 'experiment_id' in kwargs.keys() and 'ExperimentsManager' in locals():
            ExperimentsManager[kwargs['experiment_id']] = self.save_directory_name
            if ((
             'perception_save_path' in kwargs.keys() )\
             and (kwargs['perception_save_path'] is not None)
             ):
                ExperimentsManager.update_experiment(
                 kwargs['experiment_id'], 'perception_save_path',
                 kwargs['perception_save_path'])






        '''
        Create summary writer
        '''
        if not(os.path.exists(os.path.join(self.save_directory, 'summaries'))):
            os.makedirs(os.path.join(self.save_directory, 'summaries'))
        self.summary_writer = tf.summary.create_file_writer(
            os.path.join(self.save_directory, 'summaries'))
        self.summaries_directory = os.path.join(self.save_directory, 'summaries')


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
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, checkpoint_dir, max_to_keep=3)
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint is not None:
            printt("Restored from {}".format(self.ckpt_manager.latest_checkpoint),
                info=True)
        else:
            printt("Initializing from scratch.", debug=True)
        self.Model.__active_vars__.step = step_counter


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
        Start training?
        '''
        self.tb_url = None
        if 'execute' in kwargs.keys() and kwargs['execute'] is True:
            printt("Start training/testing", info=True)
            self()
    def __call__(self, *args, **kwargs):
        return self.__call__func(*args, **kwargs)

    def testing(self):
        printt("Entering testing loop", debug=True)
        epochs = 0
        step = 0
        for record_number, data_record in self.Dataset.test_dataset.enumerate():
            #diagnostics = self.Model.loss_func(data, training=False)
            #analysis_output = self.Model.analyse(diagnostics, step, self.analysis_directory)

            diagnostics, analysis_output = self.run(data_record,
                return_analysis=True,
                return_diagnostics=True, step=step)
            '''
            Print to console and save checkpoint
            '''
            print("testing epoch: %d" % epochs, end=";")
            print("data split: %d of %d" % (record_number+1, self.data_strap.dataset_length), end=";")
            print("step: %d" % steps, end=";")
            sys.stdout.write("\033[K")
            step += 1
        epochs += 1

    def run(self, input_data, return_diagnostics=True, execute_analysis=False,
        return_analysis=False, step=None, analysis_directory=None):
        '''
        Use this to run forward passes on data
        Arguments:
         - input_data : list of input data to be executed
         -
        '''

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
        tb = tb_program.TensorBoard()
        args = [None, '--logdir', self.summaries_directory, '--host', '0.0.0.0']
        if port is not None:
            args.append('--port')
            args.append(port)
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
        predict_for_input_signature_bug_run = False
        #self.Dataset.train_dataset=self.Dataset.train_dataset.skip(step)
        self.Dataset.skip(step, current_file=int(self.ckpt.current_file), epoch=int(self.ckpt.epoch))
        '''
        Start Tensorboard
        '''
        self.tensorboard()

        with self.summary_writer.as_default():
            with tf.summary.record_if(True):
                while train is True: # Trains across epochs, NOT steps
                    for record_number, data_record in\
                        self.Dataset.train_dataset.enumerate():
                            add_summary = (step+1) % self.Model.__config__.summary_steps
                            add_summary = True if add_summary == 0 else False
                            # Start Timing
                            start_time = time.time()
                            # Execute Model
                            self.Model.__update_weights__(data_record, summaries=add_summary)
                            # Print training information and duration
                            print("training epoch: {}".format(epochs+1), end=";")
                            data_split_num = record_number if self.Dataset.system_type.use_generator is False else self.Dataset.current.file
                            data_split_num_total = self.Dataset.train_dataset_steps if self.Dataset.system_type.use_generator is False else self.Dataset.num_files
                            print("data split: %d of %d" % (
                                data_split_num+1,
                                data_split_num_total), end=";")
                            print("step: %d / %d" % (step,self.Dataset.train_dataset_steps), end=";")
                            duration = time.time() - start_time
                            print("time: %.3f" % duration, end=";")
                            # Checkpointing
                            if step %\
                             self.Model.__config__.checkpoint_steps == 0:
                                self.ckpt.step.assign_add(self.Model.__config__.checkpoint_steps)
                                self.ckpt.current_file.assign(self.Dataset.current.file)
                                self.ckpt.epoch.assign(self.Dataset.current.epoch)
                                save_path = self.ckpt_manager.save()
                                print("Saved checkpoint for step {}".format(
                                    int(self.ckpt.step)), end=";")
                                sys.stdout.write("\033[K")
                            print("", end="\r")

                            # Validation
                            if step % self.Model.__config__.validation_steps == 0:
                                for validation_data_record in self.Dataset.validation_dataset.take(self.Dataset.validation_dataset_length):
                                    self.Model.loss_func(validation_data_record, training=False,
                                        validation=True, summaries=True)


                            # Save summary of the model
                            if step == 1:
                                self.Model.__forward_pass_model__.summary(print_fn=self.logging)
                                for optimisers_models in self.Model.__optimisers_models__:
                                    for model in optimisers_models['models']:
                                        model.summary(print_fn=self.logging)

                            # Increment to next step
                            step += 1
                            self.Model.__active_vars__.step = step

                    epochs += 1
                    if epochs % self.Model.__config__.saved_model_epochs == 0:
                        if not(os.path.exists(os.path.join(self.saved_model_directory, 'epoch_'+str(epochs)))):
                            os.makedirs(os.path.join(self.saved_model_directory, 'epoch_'+str(epochs)))
                        this_epoch_saved_model_dir = os.path.join(self.saved_model_directory, 'epoch_'+str(epochs))
                        saving_enabled = False if tf.__version__ == '2.0.0' else True
                        if saving_enabled is True:
                            if predict_for_input_signature_bug_run is False:
                                _ = self.Model.__forward_pass_model__.predict(data_record)
                                self.Model.__forward_pass_model__.save(this_epoch_saved_model_dir)
                                predict_for_input_signature_bug_run = True
                                print("TensorBoard started at {}".format(self.tb_url))
                            else:
                                self.Model.__forward_pass_model__.save(this_epoch_saved_model_dir)
                    if epochs >= self.Model.__config__.epochs:
                        train = False
        self.tensorboard_only()
