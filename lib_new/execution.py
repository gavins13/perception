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
with open("../Config.perception", "r") as config_file:
    Config = json.load(config_file)
from misc import printt

from contextlib import ExitStack
import os

from lib_new.model import Model

class Execution(object):
    def __init__(self, *args, **kwargs):
        '''
        Arguments:
        dataset: Dataset object (optional)
        experiment_name: string (optional)
        load_path: string (optional)
        model: Model object
        '''

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
        if 'model' in kwargs.keys() and isinstance(kwargs['model'], Model):
            # Use this dataset
            self.Model = kwargs['model']
            self.Model.create_models()


        '''
        Handle directories for saving
        '''
        project_path = Config['save_directory'] if not((
            'project_path' in kwargs['project_path'])
             and kwargs['project_path'] is not None
             ) else kwargs['project_path']

        self.experiment_name = kwargs['experiment_name'] \
            if 'experiment_name' in kwargs.keys() and \
            isinstance(kwargs['experiment_name'], str) else 'tmp'
        printt("Experiment Name: {0}".format(self.experiment_name), info=True) 
        
        datetimestr = str(datetime.now())
        datetimestr = datetimestr.replace(" ", "-")

        load_path = None if not(('load_path' in kwargs.keys()) and\
            (isinstance(kwargs['load_path'], str))) else kwargs['load_path']
        if(load_path==None):
            self.save_directory_name = experiment_name + '_' + datetimestr
            self.save_directory = os.path.join(project_path,
                self.save_directory_name)
        else:
            self.save_directory = os.path.join(project_path, load_path)
            printt("Load Dir being used.", info=True)

        if 'save_directory' in Config.keys():
            if not(os.path.exists(project_path)):
                os.makedirs(project_path)
        else:
            printt("'save_directory' attribute doesn't exist!", error=True,
                stop=True)
        self.save_directory = kwargs['save_directory']


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

        self.ckpt = tf.train.Checkpoint(**optimisers, **models, step=tf.Variable(1))
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, checkpoint_dir, max_to_keep=3)
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            printt("Restored from {}".format(self.ckpt_manager.latest_checkpoint),
                info=True)
        else:
            printt("Initializing from scratch.", info=True)


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
        if 'experiment_type' in kwargs.keys():
            exp_type = kwargs['experiment_type']
            if exp_type == 'train':
                self.__call__ = self.training
            elif exp_type == 'testing' or exp_type == 'evaluate':
                self.__call__ = self.testing
            if 'execute' in kwargs.keys() and kwargs['execute'] is True:
                self()

    def testing(self):
        epochs = 0
        step = 0
        while test is True:
            dataset_keys = self.DataFrame.get_keys() # Keys list
            for record_number, data_record in self.Dataset.train_dataset.enumerate():
                data = {key:value for i, key in enumerate(dataset_keys) for value in data_record[i]}
                diagnostics = self.Model.loss_func(training=False)
                self.Model.analyse(diagnostics, step, self.analysis_directory)
                '''
                Print to console and save checkpoint
                '''
                print("testing epoch: %d" % epochs, end=";")
                print("data split: %d of %d" % (record_number+1, self.data_strap.dataset_length), end=";")
                print("step: %d" % steps, end=";")
                print("", end='                                        \r')
                step += 1
            epochs += 1


    def training(self):
        '''
        Training Loop
        '''
        epochs = 0
        step=0
        train = True
        with self.summary_writer.as_default():
            with tf.summary_recordif(True):
                while train is True: # Trains across epochs, NOT steps
                    dataset_keys = self.DataFrame.get_keys() # Keys list
                    for record_number, data_record in\
                        self.Dataset.train_dataset.enumerate():
                            add_summary = (step+1) % self.Model.__config__.summary_steps
                            add_summary = True if add_summary == 0 else False
                            # Start Timing
                            start_time = time.time()
                            # Organise Data
                            data = {key:value for i, key in enumerate(
                                dataset_keys) for value in data_record[i]}
                            # Execute Model
                            self.Model.__update_weights__(data, summaries=add_summary)
                            # Print training information and duration
                            print("training epoch: %d" % epochs+1, end=";")
                            print("data split: %d of %d" % (
                                record_number+1,
                                self.Dataset.train_dataset_length), end=";")
                            print("step: %d" % steps, end=";")
                            duration = time.time() - start_time
                            print("time: %.3f" % duration, end=";")
                            # Checkpointing
                            self.ckpt.step.assign_add(1)
                            if int(self.ckpt.step) %\
                             self.Model.__config__.checkpoint_steps == 0:
                                save_path = manager.save()
                                print("Saved checkpoint for step {}: {}".format(
                                    int(self.ckpt.step), save_path), end=";")
                            print("",
                                end='                                        \r')

                            # Validation
                            if (step+1) % self.Model.__config__.validation_steps == 0:
                                self.Model.loss_func(training=False,
                                    validation=True, summaries=True)
                            step += 1
                    epochs += 1
                    if epochs % self.Model.__config__.saved_model_epochs == 0:
                        tf.saved_model.save(self.saved_model_directory)
                    if epochs >= self.Model.__config__.epochs:
                        train = False