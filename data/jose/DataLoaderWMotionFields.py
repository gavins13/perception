'''
Load Magnitudes cines with motion fields generated from a x16 acceleration
factor and masks

Note: returns dict: dict_keys(['u', 'v', 'gt', 'masks'])
'''

import numpy as np
#from random import shuffle, seed as __seed__
import tensorflow as tf
from __init__ import *

from scipy import stats
import pickle

#path = '/vol/biomedic/users/kgs13/PhD/projects/datasets/'
#jose_data = pickle.load(open(path+'MICCAI_cardiac_data.pkl', 'rb'), encoding='bytes')

path = '/homes/kgs13/biomedic/PhD/projects/mri_reconstruction/bin/Active Acquisition/'
motion_data = pickle.load(open(path+'results_part_7_10motionfields.pkl', 'rb'), encoding='bytes')


class Dataset(DatasetBase):
    def __init__(self, *args, **kwargs):
        super().__init__()
        '''
        Override default options
        '''
        self.dev.on = False
        self.use_direct(motion_data)
        self.num_files = None # needs setting

        '''
        Set some default properties
        '''
        self.config.batch_size = 1


        self.train_dataset_length = 7
        self.test_dataset_length = 3
        self.validation_dataset_length = 1

        self.train_dataset_steps = 7 # Should be the same as
                                        # train_data_length for batch_size
                                        # = 1, otherwise, =
                                        # train_dataset_length / batch_size
        self.test_dataset_steps = 3
        self.validation_dataset_steps = 1

        '''
        Select folds and fold number
        '''
        if 'cv_folds' in kwargs.keys():
            self.config.cv_folds = kwargs['cv_folds']
            if 'cv_fold_number' in kwargs.keys():
                self.config.cv_fold_number = kwargs['cv_fold_number']
            else:
                self.config.cv_fold_number = 1
        else:
            self.config.cv_fold_number = 1
            self.config.cv_folds = 3





    def __config__(self):
        '''
        This is executed when create() is called. It is the first method
        executed in create()
        '''
        if self.config.batch_size != 1:
            #raise ValueError('This BioBank dataset only' + \
            # ' supports batch size 1 due to images being different sizes')
            printt("Note: batching along the slice axis", warning=True)
