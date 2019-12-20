'''
class Execution:
    - load Config.perception file
    - load Experiments.perception file
    - Checkpointing, restoration, writing the saves to files
    - Keeps track of experiments by writing to a experiments txt file
    - Loads tensorboard during execution using port number specified in the exp config using no GPUs and 
'''
from lib_new.dataset import Dataset
from lib_new.model import Model
import json
with open("Config.perception", "r") as config_file:
    Config = json.load(config_file)
from lib_new.misc import printt

from contextlib import ExitStack
import os
import sys
from datetime import datetime
from lib_new.model import Model
import tensorflow as tf
import time

class Execution(object):
    def __init__(self, *args, **kwargs):
        '''
        Arguments:
        dataset: Dataset object (optional)
        experiment_name: string (optional)
        save_folder: string (optional)
        model: Model object
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

        '''
        Handle Dataset
        '''
        if 'dataset' in kwargs.keys() and isinstance(kwargs['dataset'], Dataset):
            # Use this dataset
            self.Dataset = kwargs['dataset']
        else:
            self.Dataset = Dataset()
            self.Dataset.use('developer_mode', 'cifar10')
            self.Dataset.create()

        '''
        Handle Model
        '''
        if 'model' in kwargs.keys() and issubclass(kwargs['model'], Model):
            kwargs['model'] = kwargs['model'](training=exp_type)
        if 'model' in kwargs.keys() and isinstance(kwargs['model'], Model):
            # Use this dataset
            self.Model = kwargs['model']
            self.Model.create_models()
        else:
            printt(kwargs.keys(), debug=True)
            printt(kwargs['model'], debug=True)
            printt("No Model for execution", error=True, stop=True)
        step_counter = tf.Variable(initial_value=1, trainable=False,
                                   name="global_step", dtype=tf.int64)
        self.Model.__active_vars__.step = step_counter

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

        # Create the save folder name from the experiment name above
        save_folder = None if not(('save_folder' in kwargs.keys()) and\
            (isinstance(kwargs['save_folder'], str))) else kwargs['save_folder']
        if(save_folder==None):
            self.save_directory_name = self.experiment_name + '_' + datetimestr
            self.save_directory = os.path.join(perception_save_path,
                self.save_directory_name)
        else:
            self.save_directory = os.path.join(perception_save_path, save_folder)
            printt("Load Dir being used.", info=True)
        # Create the save folder
        if not(os.path.exists(self.save_directory)):
            os.makedirs(self.save_directory)



        '''
        Create summary writer
        '''
        if not(os.path.exists(os.path.join(self.save_directory, 'summaries'))):
            os.makedirs(os.path.join(self.save_directory, 'summaries'))
        self.summary_writer = tf.summary.create_file_writer(
            os.path.join(self.save_directory, 'summaries'))


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

        self.ckpt = tf.train.Checkpoint(**optimisers, **models, step=step_counter)
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, checkpoint_dir, max_to_keep=3)
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            printt("Restored from {}".format(self.ckpt_manager.latest_checkpoint),
                info=True)
        else:
            printt("Initializing from scratch.", debug=True)


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

        '''
        Start training?
        '''
        if 'execute' in kwargs.keys() and kwargs['execute'] is True:
            printt("Start training/testing", info=True)
            self()
    def __call__(self, *args, **kwargs):
        return self.__call__func(*args, **kwargs)

    def testing(self):
        printt("Entering testing loop", debug=True)
        epochs = 0
        step = 0
        while test is True:
            for record_number, data_record in self.Dataset.train_dataset.enumerate():
                #diagnostics = self.Model.loss_func(data, training=False)
                #analysis_output = self.Model.analyse(diagnostics, step, self.analysis_directory)

                diagnostics, analysis_output = self.run(data_record,
                    return_analysis=True,
                    return_diagnostics=True)
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
        '''

        analysis_directory = self.analysis_directory if analysis_directory is None else analysis_directory

        execute_analysis = True if return_analysis is True else execute_analysis

        diagnostics = self.Model.loss_func(data, training=False)
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
        

    def training(self):
        '''
        Training Loop
        '''
        printt("Entering training loop", debug=True)
        epochs = 0
        step=0
        train = True
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
                            print("data split: %d of %d" % (
                                record_number+1,
                                self.Dataset.train_dataset_steps), end=";")
                            print("step: %d" % step, end=";")
                            duration = time.time() - start_time
                            print("time: %.3f" % duration, end=";")
                            # Checkpointing
                            self.ckpt.step.assign_add(1)
                            if int(self.ckpt.step) %\
                             self.Model.__config__.checkpoint_steps == 0:
                                save_path = self.ckpt_manager.save()
                                print("Saved checkpoint for step {}: {}".format(
                                    int(self.ckpt.step), save_path), end=";")
                                sys.stdout.write("\033[K")
                            print("", end="\r")

                            # Validation
                            if (step+1) % self.Model.__config__.validation_steps == 0:
                                self.Model.loss_func(data_record, training=False,
                                    validation=True, summaries=True)
                            step += 1
                    epochs += 1
                    if epochs % self.Model.__config__.saved_model_epochs == 0:                        
                        if not(os.path.exists(os.path.join(self.saved_model_directory, 'epoch_'+str(epochs)))):
                            os.makedirs(os.path.join(self.saved_model_directory, 'epoch_'+str(epochs)))
                        this_epoch_saved_model_dir = os.path.join(self.saved_model_directory, 'epoch_'+str(epochs))
                        tf.saved_model.save(self.Model.__forward_pass_model__, this_epoch_saved_model_dir)
                    if epochs >= self.Model.__config__.epochs:
                        train = False