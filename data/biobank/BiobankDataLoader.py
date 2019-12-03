import json
import sys
#sys.path.insert(0, '../lib/data_frame.py')
#from data_frame import Dataset
import numpy as np
import nibabel
from os import path
from random import shuffle, seed as __seed__
import tensorflow as tf

biobank_list_path = 'biobank.json'
basepath = path.dirname(__file__)
biobank_list_path = path.abspath(path.join(basepath, biobank_list_path))

def printt(val, warning=False):
    print(val)


class Dataset():
    counter=0 # counter for threads
    def __init__(self):
        #self.counter = 0
        self.gen_num = Dataset.counter
        Dataset.counter += 1
        class Config: pass
        self.system_type = Config()
        self.system_type.use_generator = False # Use TF Generator
        self.system_type.use_direct = False # Load all into memory

        self.dev = Config()
        self.dev.on = True
        self.dev.dataset = 'cifar10'

        self.generator = Config()
        self.generator.data_types = None

        self.config = Config()
        self.config.type = 'train' # 'train', 'test', 'validate'
        self.config.batch_size = None
        self.config.prefetch_factor = 4
        self.config.epochs = None

        '''
        Only valid when the using direct method
        '''
        self.config.dataset_split = [60,30,10] # Train, Test, Validate
        self.config.cv_folds = None  
        self.config.cv_fold_number = None
        self.config.validation_size = 1

        self.current = Config()
        self.current.epoch = -1
        self.current.step = 0
        self.current.file = 0
    def use_generator(self, data_types):
        self.system_type.use_generator = True
        self.system_type.use_direct = False
        self.generator_data_types = data_types
        self.dev.on = False
    def use_direct(self, dataset):
        self.system_type.use_direct = True
        self.system_type.use_generator = False
        self.dev.on = False
        raise NotImplementedError()
    def use_developer_mode(self):
        self.dev.on = True

    def py_gen(self, gen_name):
        # This function will basically return the Train data and Validation record if
        gen_name = gen_name.decode('utf8') + '_' + str(self.gen_num)
        for num in range(5):
            #sleep(0.3)
            yield '{} yields {}'.format(gen_name, num)
    def py_gen_train(self, gen_name):
        pass
    def py_gen_test(self, gen_name):
        pass
    def py_gen_validation(self, gen_name):
        pass

    def __config__(self):
        pass

    def create(self, threads=4):
        self.set_operation_seed()
        self.__config__()
        '''
        threads = Number of computational threads to perform the retrieval of data
        '''
        if self.dev.on is True:
            if self.dev.dataset == 'cifar10':
                self.cifar10()
            else:
                raise NotImplementedError()
        elif self.dev.on is False:
            if self.system_type.use_generator is True:
                self.Datasets = [
                    self.create_generator('py_gen_train', threads=threads),
                    self.create_generator('py_gen_test', threads=threads),
                    self.create_generator('py_gen_validation', threads=threads)
                ]
                self.Dataset = self.Datasets[0]
            elif self.system_type.use_direct is True:
                raise NotImplementedError()
        self.process_dataset()
    def create_generator(self, generator_name='py_gen_train', threads=4):
        dummy_ds = tf.data.Dataset.from_tensor_slices(['DataGenerator']*threads)
        dummy_ds = dummy_ds.interleave(
            lambda x: tf.data.Dataset.from_generator(
                #lambda x: self.__class__().py_gen(x),
                lambda x: getattr(self, generator_name)(x),
                output_types=self.generator_data_types, args=(x,)
            ),
            cycle_length=threads,
            block_length=1,
            num_parallel_calls=threads)
        return dummy_ds
    def process_dataset(self):
        '''
        For direct: Creates validation folds
        For generator: access the three generators neccesssary
        '''
        if self.config.batch_size is None:
            printt("Batch size is not set so Default is set to 1", warning=True)
            self.config.batch_size = 1
        if self.system_type.use_direct is True or self.dev.on is True:
            buffer_size = None if not(hasattr(self, 'buffer_size')) else getattr(self, 'buffer_size')
            self.Dataset = self.Dataset.shuffle(buffer_size=buffer_size, seed=operation_seed)
            if self.config.cv_folds is not None:
                self.Datasets = []
                if self.config.cv_fold_number is None:
                    raise ValueError('Invalid fold number specified: ', self.config.cv_fold_number)
                train_ds = []
                for i in range(self.config.cv_folds):
                    tmp = self.Dataset.shard(self.config.cv_folds, i)
                    if self.config.cv_fold_number == i:
                        test_ds = tmp
                    else:
                        train_ds.append(tmp)
                first = train_ds[0]
                del(train_ds[0])
                for x in train_ds:
                    first = first.concatenate(x)
                train_ds = first
                validation_ds = test_ds.take(self.config.validation_size)
                self.Datasets = [train_ds, test_ds, validation_ds]
                self.Datasets = [x(batch_size=self.config.batch_size) for x in self.Datasets]
                self.Datasets = [x(buffer_size=self.config.batch_size*self.config.prefetch_factor) for x in self.Datasets]
            elif self.config.dataset_split is not None:
                raise NotImplementedError('Dataset splitting not implemented yet')
        else:
            #self.Datasets = [x.batch(batch_size=self.config.batch_size) for x in self.Datasets]
            self.Datasets = [x.prefetch(buffer_size=self.config.batch_size) for x in self.Datasets]
            #self.Dataset = self.Dataset.batch(batch_size=self.config.batch_size)
            #self.Dataset = self.Dataset.prefetch(buffer_size=self.config.batch_size*self.config.prefetch_factor)

    def cifar10(self):
        # Load training and eval data from tf.keras
        (train_data, train_labels), _ = tf.keras.datasets.cifar10.load_data()

        train_data = train_data.reshape(-1, 32, 32, 3).astype('float32')
        train_data = train_data / 255.
        train_labels = np.asarray(train_labels, dtype=np.int32)

        train_data = train_data[0:49920]
        train_labels = train_labels[0:49920]


        tf.random.set_seed(69)
        operation_seed = None

        # for train
        self.Dataset = tf.data.Dataset.from_tensor_slices(train_data, train_labels)




    def set_operation_seed(self, seed=1114):
        np.random.seed(seed)
        __seed__(1114)



class BioBank(Dataset):
    def import_list(self):
        self.file_list = json.load(open(biobank_list_path, 'rb'))
    def shuffle_and_split(self, cv_folds=3, cv_fold_num=1):
        n = len(self.file_list)
        #self.file_list = np.random.shuffle(self.file_list)
        shuffle(self.file_list)
        n_ = np.int(np.float(n)/np.float(cv_folds))
        fold_remainder = n-((cv_folds-1)*n_)
        fold_sizes = [n_]*(cv_folds-1) + [fold_remainder]
        self.train_file_list = self.file_list[0:fold_sizes[0]]
        self.test_file_list = self.file_list[fold_sizes[0]:fold_sizes[0]+fold_sizes[1]]
        self.validation_file_list = [self.test_file_list[0]]

    def __config__(self):
        '''
        This is executed when create() is called. It is the first method
        executed in create()
        '''
        if self.config.batch_size != 1:
            #raise ValueError('This BioBank dataset only' + \
            # ' supports batch size 1 due to images being different sizes')
            printt("Note: batching along the slice axis", warning=True)
        self.import_list()
        self.shuffle_and_split()
    def py_gen(self, gen_name):
        raise NotImplemented("You're probably using the wrong function")
        # This function will basically return the Train data and Validation record if
        gen_name = gen_name.decode('utf8') + '_' + str(self.gen_num)
        for num in range(5):
            #sleep(0.3)
            yield '{} yields {}'.format(gen_name, num)
    def py_gen_train(self, gen_name):
        gen_name = gen_name.decode('utf8') + '_' + str(self.gen_num)
        epochs = -999 if self.config.epochs is None else self.config.epochs

        self.current.epoch = -1
        self.current.step = 0
        self.current.file = -1

        current_batch_max = None
        current_batch_size = None
        current_num_slices = 0
        while self.current.epoch != epochs+1:
            if (current_batch_max is None) or (current_num_slices >= current_batch_max):
                # LOAD NEXT FILE
                #print("Epoch %d - File %d of %d                    " % (self.current.epoch, self.current.file, len(self.train_file_list)), end="\r")
                current_num_slices = 0
                self.current.file += 1
                idx = self.current.file % len(self.train_file_list)
                filename = self.train_file_list[self.current.file]
                data = nibabel.load(filename)
                d = data.get_fdata() # numpy data # [H, W, SLICE, TIME] # (210, 208, 11, 50) 
                current_batch_max = d.shape[2]
                d = np.transpose(d, [0,1,3,2]) # [H, W, TIME, SLICE]
                current_batch_size = self.config.batch_size if self.config.batch_size < current_batch_max else current_batch_max
            #else:
            #    # CURRENT FILE
            #    pass

            this_batch_size = current_batch_size if (current_batch_max-current_num_slices) >  current_batch_size else (current_batch_max-current_num_slices)
            this_data = d[:,:,:,current_num_slices:current_num_slices+this_batch_size]
            #print(this_batch_size)
            #print(current_batch_size)
            #print(current_batch_max)
            #print(this_data.shape)
            #this_data = np.squeeze(this_data, axis=3)
            # [] to delete this line
            #this_data = this_data[:,:,0:3]
            # [] end delete
            current_num_slices += this_batch_size
            self.current.step += 1
            if idx == 0:
                self.current.epoch += 1
            yield tf.convert_to_tensor(this_data)


    def py_gen_test(self, gen_name):
        gen_name = gen_name.decode('utf8') + '_' + str(self.gen_num)
        epochs = -999 if self.config.epochs is None else self.config.epochs

        current_epoch = -1
        current_step = 0
        current_file = -1

        current_batch_max = None
        current_batch_size = None
        current_num_slices = 0
        while current_epoch != epochs+1:
            if current_num_slices >= current_batch_max:
                # LOAD NEXT FILE
                current_file += 1
                idx = current_file % len(self.test_file_list)
                filename = self.train_file_list[current_file]
                data = nibabel.load(filename)
                d = data.get_fdata() # numpy data # [H, W, SLICE, TIME] # (210, 208, 11, 50) 
                current_batch_max = d.shape[2]
                d = np.transpose(d, [0,1,3,2]) # [H, W, TIME, SLICE]
                current_batch_size = self.config.batch_size if self.config.batch_size < current_batch_max else current_batch_max
            #else:
            #    # CURRENT FILE
            #    pass

            this_batch_size = current_batch_size if (current_batch_max-current_num_slices) > current_batch_size else  (current_batch_max-current_num_slices)
            this_data = d[:,:,:,current_num_slices:current_num_slices+this_batch_size]
            current_num_slices += this_batch_size
            current_step += 1
            if idx == 0:
                current_epoch += 1
            yield this_data
    def py_gen_validation(self, gen_name):
        gen_name = gen_name.decode('utf8') + '_' + str(self.gen_num)
        epochs = -999 if self.config.epochs is None else self.config.epochs

        current_epoch = -1
        current_step = 0
        current_file = -1

        current_batch_max = None
        current_batch_size = None
        current_num_slices = 0
        while current_epoch != epochs+1:
            if current_num_slices >= current_batch_max:
                # LOAD NEXT FILE
                current_file += 1
                idx = current_file % len(self.validation_file_list)
                filename = self.train_file_list[current_file]
                data = nibabel.load(filename)
                d = data.get_fdata() # numpy data # [H, W, SLICE, TIME] # (210, 208, 11, 50) 
                current_batch_max = d.shape[2]
                d = np.transpose(d, [0,1,3,2]) # [H, W, TIME, SLICE]
                current_batch_size = self.config.batch_size if self.config.batch_size < batch_max else batch_max
            #else:
            #    # CURRENT FILE
            #    pass

            this_batch_size = current_batch_size if (current_batch_max-current_num_slices) >current_batch_size else (current_batch_max-current_num_slices)
            this_data = d[:,:,:,no_slices:no_slices+this_batch_size]
            no_slices += this_batch_size
            current_step += 1
            if idx == 0:
                current_epoch += 1
            yield this_data
