import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from __init__ import *

from scipy import stats
import pickle
import sys
perception_path = '/vol/biomedic/users/kgs13/PhD/projects/'
sys.path.insert(0, perception_path)
#import perception_tf2 as perception
printt = perception.printt
DatasetBase = perception.Dataset
#path = '/rds/general/user/kgs13/home/_Home/PhD/projects/datasets/jose/'
#jose_data = pickle.load(open(path+'MICCAI_cardiac_data.pkl', 'rb'), encoding='bytes')

path = '/vol/medic02/users/kgs13/datasets/jose/'
jose_data = pickle.load(open(path+'cardiac_cine_mri_data.pkl', 'rb'), encoding='bytes')

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

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise


@tf.function
def add_noise(image, std=1.):
    #image = tf.transpose(image, [0,2,3,1]) # [B, H, W, T]
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=std, dtype=tf.float64, seed=1117)
    noise_2 = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=std, dtype=tf.float64, seed=1118)
    complex_noise = tf.complex(noise, noise_2)
    #return tf.transpose(image + noise, [0,3,1,2])
    return image+complex_noise

class Dataset(DatasetBase):
    def __init__(self, *args, **kwargs):
        '''
        Options:

        biobank_test_dataset: (bool) Set to True if you wish to access the
                                     reduced BioBank test dataset
        '''
        kwargs['cv_folds'] = 3
        super().__init__(*args, **kwargs)
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

        self.validation_dataset_length = 1
        self.validation_dataset_steps = 1


        # Start Customising class
        class Config: pass
        self.config.patches = Config()
        self.config.patches.enabled = True
        self.config.patches.size = 32
        if 'patches' in kwargs.keys() and isinstance(kwargs['patches'], bool):
            self.config.patches.enabled = kwargs['patches']
        if 'patch_size' in kwargs.keys() and isinstance(kwargs['patch_size'], int):
            self.config.patches.size = kwargs['patch_size']

        self.config.biobank_test_dataset = (
            'biobank_test_dataset' in kwargs.keys() and \
                 isinstance(kwargs['biobank_test_dataset'], bool)
            ) and kwargs['biobank_test_dataset']

        self.config.add_noise = (
            'add_test_dataset_noise' in kwargs.keys() and \
                 isinstance(kwargs['add_test_dataset_noise'], bool)
            ) and kwargs['add_test_dataset_noise']

        
        self.config.noise = self.module_arg('noise_val', false_val=0.02)
        tf.random.set_seed(1114)
        self.add_noise_func = lambda image : add_noise(image, std=self.config.noise)

        self.config.patches.sparse = self.module_arg('sparse_patches', false_val=False)
        
    def __process_dataset__(self):
        super().__process_dataset__()
        # Augmentation step
        self.train_dataset=self.train_dataset.map(rotate_tf)
        if self.config.patches.enabled is True:
            printt("Patches enabled", info=True)
            #patch_positions = [0,30,45,60,75,90, 95]  +  list(range(100, 122, 2)) + [130, 135,150,165,180,194,224]
            if self.config.patches.sparse is False:
                patch_positions = [35,30,45,60,75,90, 95]  +  list(range(100, 122, 2)) + [130, 135,150,165,180,194,189]
            else:
                printt("Using sparse patches!", info=True)
                #patch_positions = list(range(35, 195, 20))
                patch_positions = [35, 55, 75, 95, 115, 120, 125, 135, 140, 145, 155, 175, 195, 215]
            patch_size = self.config.patches.size
            init_map  = lambda d, x : d[:,:,:,x:x+patch_size]
            end_map = lambda d: tf.concat([init_map(d, x) for x in patch_positions], axis=0)
            #self.train_dataset = self.train_dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(end_map(x)))
            self.train_dataset = self.train_dataset.map(end_map)
            self.train_dataset = self.train_dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
            self.train_dataset = self.train_dataset.shuffle(self.train_dataset_length * len(patch_positions), seed=1114)
            self.train_dataset = self.train_dataset.batch(self.config.batch_size)
            #self.train_dataset = self.train_dataset.map(end_map)
            self.train_dataset_length = int(self.train_dataset_length * len(patch_positions))

        if self.config.biobank_test_dataset is True:
            printt("Using Biobank Test Dataset!!", info=True)
            self.test_dataset = tf.data.Dataset.from_tensor_slices((biobank_data))
            self.test_dataset = self.test_dataset.shuffle(
                buffer_size=biobank_data.shape[0],
                seed=1114
            )
            b = self.get_batch_sizes()
            self.test_dataset = self.test_dataset.batch(batch_size=b[2])
            self.test_dataset_length = biobank_data.shape[0]

        if self.config.add_noise is True:
            self.test_dataset = self.test_dataset.map(self.add_noise_func, deterministic=True)

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
