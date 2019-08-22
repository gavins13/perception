import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt

import nibabel as nib
from random import shuffle

import pickle
import numpy as np

import sys
import os
import collections
from scipy.ndimage.interpolation import map_coordinates, shift as translate_img, rotate as rotate_img
from scipy.ndimage.filters import gaussian_filter

import itertools
#sys.path.insert(0, '../')
from mri_scanner import MRImageEditor as MRI

ExtraData = collections.namedtuple('ExtraData', ('k_space_full',
                                                 'k_space_masked',
                                                 'image_data_complex',
                                                 'k_space_mask'))
ExtraDataSet = collections.namedtuple('ExtraDataSet', ('train', 'test'))


class load_data(object):
    def __init__(self, dataset="acc=0.20"):
        self.dataset_folder = dataset
        self.extra_data = True

    def __call__(self, tr=None, te=None, acc_factor=0.20, save_path='/vol/biomedic/users/kgs13/PhD/projects/datasets/jose/saved_datasets/', load_from_saved=True):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        save_path_pkl = save_path+self.dataset_folder+'/'
        if load_from_saved is True:
            print("Load from saved...")
            to_load = save_path_pkl + "/dataset.p"
            print("Loading from %s" % to_load)
            dataset = pickle.load(open(to_load, "rb"))
            print("Finished load from saved...")
            return dataset

        data_folders = ['']
        path = '/vol/bitbucket/js3611/caffe/examples/SegCSCNN/data/cardiac/train/3of3/'
        data = pickle.load(open(path+'train_0.pkl', 'rb'), encoding='bytes')
        dataset = data[b'data']
        self.dataset_shape = list(dataset.shape)  # [10, 30, 256, 256]


        Scanner = MRI(256, 256)

        def scan(train):
            train = self.reduce_time_dims(train)
            train_labels = []
            train_kspace_gt = []
            train_kspace_masked = []
            train_kspace_mask = []
            for i in range(train.shape[0]):
                print("Scan %d of %d" % (i, train.shape[0]))
                Scanner.set_mask_v2(acc_factor)
                label, kspace_gt, kspace_masked = Scanner.mask_image(train[i,:,:], return_kspaces=True, return_complex=True)
                train_labels.append(label)
                train_kspace_gt.append(kspace_gt)
                train_kspace_masked.append(kspace_masked)
                train_kspace_mask.append(Scanner.mask)
            train_labels = self.expand_time_dims(np.asarray(train_labels))
            train_kspace_gt = self.expand_time_dims(np.asarray(train_kspace_gt))
            train_kspace_masked = self.expand_time_dims(np.asarray(train_kspace_masked))
            train_kspace_mask = self.expand_time_dims(np.asarray(train_kspace_mask))
            return train_labels, train_kspace_gt, train_kspace_masked, train_kspace_mask

        dataset = np.abs(dataset) # since dataset is originally complex
        # start pseudocode
        train = dataset[0:7, :,:,:]
        #train = self.augment_cines(train)
        self.dataset_shape[0]=train.shape[0]
        #train_labels,train_kspace_gt,train_kspace_masked,train_kspace_mask = scan(train)
        train_list = scan(train)
        train = train.astype('complex128')
        self.dataset_shape[0]=10

        test = dataset[7:10, :,:,:]
        #test = self.augment_cines(test)
        self.dataset_shape[0]=test.shape[0]
        #test_labels,test_kspace_gt,test_kspace_masked,test_kspace_mask = scan(test)
        test_list = scan(test)
        test = test.astype('complex128')
        self.dataset_shape[0]=10


        train_list = [self.reduce_spatial_dims(x) for x in train_list]
        test_list = [self.reduce_spatial_dims(x) for x in test_list]
        train = self.reduce_spatial_dims(train)
        test = self.reduce_spatial_dims(test)

        extra_data = ExtraData(ExtraDataSet(train_list[1], test_list[1]), ExtraDataSet(train_list[2], test_list[2]), ExtraDataSet(train_list[0], test_list[0]), ExtraDataSet(train_list[3], test_list[3]))

        dataset = [train_list[0], train, test_list[0], test, extra_data]
        if(save_path!=None):
            os.mkdir(save_path_pkl)
            pickle.dump(dataset, open(save_path_pkl + "/dataset.p", "wb"))
        return dataset


    def reduce_spatial_dims(self, data):
        # receive [sample, time, spatial_1, spatial_2]
        # output [sample, time, spatial_1*spatial_2]
        this_shape = list(data.shape)
        data = np.reshape(data, this_shape[0:2] +
                          [this_shape[2]*this_shape[3], 1])
        data = np.squeeze(data, axis=3)
        return data

    def expand_spatial_dims(self, data):
        # receive [sample, time, spatial_1*spatial_2]
        # output [sample, time, spatial_1, spatial_2]
        this_shape = list(data.shape)
        data = np.expand_dims(data, axis=3)
        data = np.reshape(data, this_shape[0:2] +
                          [self.dataset_shape[2], self.dataset_shape[3]])
        return data

    def reduce_time_dims(self, data):
        # receive [sample, time, spatial_1, spatial_2]
        # output [sample*time, spatial_1, spatial_2]
        this_shape = list(data.shape)
        data = np.transpose(data, [2, 3, 0, 1])
        data = np.reshape(data, this_shape[2::] +
                          [this_shape[0]*this_shape[1], 1])
        data = np.squeeze(data, axis=3)
        data = np.transpose(data, [2, 0, 1])
        return data

    def expand_time_dims(self, data):
        # receive [sample*time, spatial_1, spatial_2]
        # output [sample, time, spatial_1, spatial_2]
        this_shape = list(data.shape)
        data = np.transpose(data, [1, 2, 0])
        data = np.expand_dims(data, axis=3)
        data = np.reshape(data, this_shape[1:3] + [self.dataset_shape[0],
                                                   self.dataset_shape[1]])
        data = np.transpose(data, [2, 3, 0, 1])
        return data

    def augment_cines(self, cines):
        # expect [N, Nt, x, y]
        aug_cines = []
        for i in range(cines.shape[0]):
            print("Augmented cine %d of %d" % (i, cines.shape[0]))
            aug_cines.append(self.augment_cine(cines[i,:,:,:]))
        aug_cines = np.concatenate(aug_cines, axis=0)
        return aug_cines # [N*N_aug, Nt, x, y]

    def augment_cine(self, cine):
        # expect [Nt, x, y]
        # traslate +- 20 pixels
        # rotate 0, 2pi
        # reflection on spatial AND temporal axis
        # elastic deformation with parameters alpha=[0->3], sigma=[0.05->0.1]
        x = np.linspace(-20., 20., 10).tolist()
        y = np.linspace(-20., 20., 10).tolist()
        theta = np.linspace(0., 2.*np.pi, 20).tolist()
        alpha = np.linspace(0., 3., 10).tolist()
        sigma = np.linspace(0.05, 0.1, 10).tolist()
        reflect_space = [0, 1]
        reflect_time = [0, 1]
        keys = 'x', 'y', 'theta', 'alpha', 'sigma', 'reflect_space_x','reflect_space_y', 'reflect_time'
        # Find combinations of all:
        #combinations = list(itertools.product(*[x, y, theta, alpha, sigma]))
        augmentations = [dict(zip(keys, combo)) for combo in itertools.product(
                        *[x, y, theta, alpha, sigma, reflect_space,
                          reflect_space, reflect_time])]
        dataset = []
        for augmentation in augmentations:
            frames = []
            for frame_i in range(cine.shape[0]):
                this_frame = cine[frame_i, :, :]
                transformed = self.elastic_transform(this_frame,
                                                     augmentation["alpha"],
                                                     augmentation["sigma"])
                transformed = translate_img(transformed, [augmentation["x"],
                                                          augmentation["y"]])
                transformed = rotate_img(transformed, augmentation["theta"])
                if(augmentation["reflect_space_x"] == 1):
                    transformed = np.flip(transformed, axis=0)
                if(augmentation["reflect_space_y"] == 1):
                    transformed = np.flip(transformed, axis=1)
                frames.append(transformed)
            frames = np.asarray(frames)
            if(augmentation["reflect_time"] == 1):
                frames = np.flip(frames, axis=0)
            dataset.append(frames)
        dataset = np.asarray(dataset)
        return dataset



    @staticmethod
    def elastic_transform(image, alpha, sigma, random_state=None):
        image = np.expand_dims(image, axis=2)
        """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.
        """
        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

        distored_image = map_coordinates(image, indices, order=1, mode='reflect')
        return np.squeeze(distored_image.reshape(image.shape), axis=2)
