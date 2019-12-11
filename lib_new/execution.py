
class Execution:
    - load Config.perception file
    - load Experiments.perception file
    - Checkpointing, restoration, writing the saves to files
    - Keeps track of experiments by writing to a experiments txt file
    - Loads tensorboard during execution using port number specified in the exp config using no GPUs and 

from dataset import Dataset
from learning import Model
import json
with open("../Config.perception", "r") as config_file:
    Config = json.load(config_file)
from misc import printt

from contextlib import ExitStack
import os

class Execution(object):
    def __init__(self, *args, **kwargs):
        '''
        Arguments:
        dataset: Dataset object (optional)
        experiment_name: string (optional)
        load: string (optional)
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
        Handle directories for saving
        '''
        project_path = Config['save_directory']

        self.experiment_name = kwargs['experiment_name'] \
            if 'experiment_name' in kwargs.keys() and \
            isinstance(kwargs['experiment_name'], str) else 'tmp'
        printt("Experiment Name: {0}".format(self.experiment_name), info=True) 
        
        datetimestr = str(datetime.now())
        datetimestr = datetimestr.replace(" ", "-")

        load = None if not(('load' in kwargs.keys()) and\
            (isinstance(kwargs['load'], str))) else kwargs['load']
        if(load==None):
            self.save_directory_name = experiment_name + '_' + datetimestr
            self.save_directory = os.path.join(project_path,
                self.save_directory_name)
        else:
            self.save_directory = os.path.join(project_path, load)
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
        if not(os.path.exists(os.path.join(project_path, 'summaries'))):
            os.makedirs(os.path.join(project_path, 'summaries'))
        self.summary_writer = tf.summary.create_file_writer(
            os.path.join(project_path, 'summaries'))


        '''
        Create checkpointer and restore
        '''
        if not(os.path.exists(os.path.join(project_path, 'checkpoints'))):
            os.makedirs(os.path.join(project_path, 'checkpoints'))        
        checkpoint_dir = os.path.join(project_path, 'checkpoints')
        optimisers = {'opt_'+str(i): opt for i,opt in enumerate(
            self.Model.__optimisers__)}

        ckpt = tf.train.Checkpoint(**optimisers)
        self.ckpt_manager = tf.train.CheckpointManager(
            ckpt, checkpoint_dir, max_to_keep=3)
        ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            printt("Restored from {}".format(self.ckpt_manager.latest_checkpoint),
                info=True)
        else:
            printt("Initializing from scratch.", info=True)

    def testing(self):
        epochs = 0
        while test is True:
            dataset_keys = self.DataFrame.get_keys() # Keys list
            for record_number, data_record in self.Dataset.train_dataset.enumerate():
                data = {key:value for i, key in enumerate(dataset_keys) for value in data_record[i]}
                diagnostics = self.Model.loss_func(training=False)
                '''
                Print to console and save checkpoint
                '''
                print("testing epoch: %d" % epochs, end=";")
                print("data split: %d of %d" % (record_number+1, self.data_strap.dataset_length), end=";")
                print("step: %d" % steps, end=";")
                ckpt.step.assign_add(1)
                if int(ckpt.step) % checkpoint_steps == 0:
                  save_path = manager.save()
                  print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path), end=";")
                print("", end='                                        \r')
            epochs += 1


    def training(self):
        '''
        Training Loop
        '''
        epochs = 0
        step=0
        with self.summary_writer.as_default():
            with tf.summary_recordif(True):
                while train is True: # Trains across epochs, NOT steps
                    dataset_keys = self.DataFrame.get_keys() # Keys list
                    for record_number, data_record in self.Dataset.train_dataset.enumerate():
                        start_time = time.time()
                        data = {key:value for i, key in enumerate(dataset_keys) for value in data_record[i]}
                        self.Model.__update_weights__(data)
                        print("training epoch: %d" % epochs+1, end=";")
                        print("data split: %d of %d" % (record_number+1, self.Dataset.train_dataset_length), end=";")
                        print("step: %d" % steps, end=";")
                        duration = time.time() - start_time
                        print("time: %.3f" % duration, end=";")
                        ckpt.step.assign_add(1)
                        if int(ckpt.step) % checkpoint_steps == 0:
                          save_path = manager.save()
                          print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path), end=";")
                        print("", end='                                        \r')
                        step += 1
                    epochs += 1
                    
            # saving (checkpoint) the model every save_epochs
            if (epoch + 1) % save_epochs == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)