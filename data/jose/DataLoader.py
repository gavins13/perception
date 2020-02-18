UNFINISHED

import numpy as np
from random import shuffle, seed as __seed__
import tensorflow as tf
from lib.misc import printt
from lib.dataset import Dataset as DatasetBase
from scipy import stats
import pickle

path = '/vol/biomedic/users/kgs13/PhD/projects/datasets/'
jose_data = pickle.load(open(path+'MICCAI_cardiac_data.pkl', 'rb'), encoding='bytes')

# This dataset is simple a (10, 30, 256, 256) complex128 matrix

class Dataset(DatasetBase):
    def __init__(self, *args, **kwargs):
        super().__init__()
        '''
        Override default options
        '''
        self.dev.on = False
        self.use_direct(jose_data)
        self.num_files = None # needs setting

        '''
        Set some default properties
        '''
        self.config.batch_size = 1

        '''
        Select folds and fold number
        '''
        if 'cv_folds' in kwargs.keys():
            self.config.cv_folds = kwargs['cv_folds']
            if 'cv_fold_number' in kwargs.keys():
                self.config.cv_fold_num = kwargs['cv_fold_number']
            else:
                self.config.cv_fold_num = 1
        else:
            self.config.cv_fold_num = 1
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