import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from __init__ import *

from scipy import stats
import pickle

path = '/vol/biomedic/users/kgs13/PhD/projects/datasets/'
jose_data = pickle.load(open(path+'MICCAI_cardiac_data.pkl', 'rb'), encoding='bytes')

# This dataset is simple a (10, 30, 256, 256) complex128 matrix

@tf.function
def rotate_tf(image):
    image = tf.transpose(image, [0,2,3,1])
    if image.shape.__len__() ==4:
        random_angles = tf.random.uniform(shape = (tf.shape(image)[0], ), minval = -np
        .pi / 2, maxval = np.pi / 2)

        random_x = tf.random.uniform(shape = (tf.shape(image)[0], 1), minval = -16, maxval = 16)
        random_y = tf.random.uniform(shape = (tf.shape(image)[0], 1), minval = -32, maxval = 32)
        translate = tf.concat([random_y, random_x], axis=1)
    if image.shape.__len__() ==3:
        random_angles = tf.random.uniform(shape = (), minval = -np
        .pi / 2, maxval = np.pi / 2)
    real = tfa.image.rotate(tf.math.real(image),random_angles, interpolation="BILINEAR")
    real = tfa.image.translate(real, translate, interpolation="NEAREST")
    imag = tfa.image.rotate(tf.math.imag(image),random_angles, interpolation="BILINEAR")
    imag = tfa.image.translate(imag, translate, interpolation="NEAREST")
    rotated = tf.complex(real, imag)
    return tf.transpose(rotated, [0,3,1,2])

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

        # Start Customising class
        class Config: pass
        self.config.patches = Config()
        self.config.patches.enabled = True
        self.config.patches.size = 32
        if 'patches' in kwargs.keys() and isinstance(kwargs['patches'], bool):
            self.config.patches.enabled = kwargs['patches']
        if 'patch_size' in kwargs.keys() and isinstance(kwargs['patch_size'], int):
            self.config.patches.size = kwargs['patch_size']


    def __process_dataset__(self):
        super().__process_dataset__()
        # Augmentation step
        self.train_dataset=self.train_dataset.map(rotate_tf)

        #patch_positions = [0,30,45,60,75,90, 95]  +  list(range(100, 122, 2)) + [130, 135,150,165,180,194,224]
        patch_positions = [35,30,45,60,75,90, 95]  +  list(range(100, 122, 2)) + [130, 135,150,165,180,194,189]
        patch_size = self.config.patches.size
        init_map  = lambda d, x : d[:,:,:,x:x+patch_size]
        end_map = lambda d: tf.concat([init_map(d, x) for x in patch_positions], axis=0)
        self.train_dataset = self.train_dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(end_map(x)))
        self.train_dataset = self.train_dataset.shuffle(self.train_dataset_length * len(patch_positions), seed=1114)
        self.train_dataset = self.train_dataset.batch(self.config.batch_size)
        #self.train_dataset = self.train_dataset.map(end_map)
        self.train_dataset_length = int(self.train_dataset_length * len(patch_positions))
        self.set_dataset_steps()


    def skip(self, steps, current_file=None, epoch=None):
        '''
        The Dataset API of TensorFlow has a .skip() method that doesn't work
        with the random seeds. Hence do nothing on .skip()
        '''
        pass

    def __config__(self):
        '''
        This is executed when create() is called. It is the first method
        executed in create()
        '''
        if self.config.batch_size != 1:
            #raise ValueError('This BioBank dataset only' + \
            # ' supports batch size 1 due to images being different sizes')
            printt("Note: batching along the slice axis", warning=True)
