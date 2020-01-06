import numpy as np
from random import shuffle, seed as __seed__
import tensorflow as tf
from .misc import printt


class Dataset():
    '''
    The Dataset class provides the abstraction that allows:
        1. Automatic handling of training, testing and validation splitting
        2. Dataset shuffling (with automatic seed setting)
        3. Prefetching and multi-threading for Python generator functions
        4. Automatic test mode

    Usage:
    To use, simply initialise, call use(), then create().
    Afer this, self.Datasets is a list of 3 items: the training, testing and
    validation datasets that you can loop through during training/testing/etc.

    Additionally, after inheritance of Dataset, you can invoke use() in the
    __init__ method (which will need to call super().__init__())

    Notes:
    Before, create(), might be worth also setting config.cv_folds and
    config.cv_folds_number for the number of folds and which fold to operate 
    on for the current running of Perception. Note: cross-validation folds are
    generated after during the create() calling, either through __config__()
    for the generator mode, or in the __process_dataset__() method for direct/
    developer mode. 

    Generator mode:
    When the generator mode is used, and the generator is specified, remember
    to specify in the inherited 'generator' class a '__config__' method which
    will split the generator sources (e.g. files) into 'self.config.folds'
    number of folds, and then also select a fold that the class' Python
    generators can use.

    Remember to define self.train_dataset_length, self.test_dataset_length
    and self.validation_dataset_length in the class

    Developer mode:
    If using developer mode, remember to set self.dev.dataset before calling
    use()

    '''
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
        self.num_files = None

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
        self.current.epoch = None # -1
        self.current.step = None
        self.current.file = -1 # -1

        self.train_dataset_length = None
        self.test_dataset_length = None
        self.validation_dataset_length = None

        self.train_dataset_steps = None # Should be the same as
                                        # train_data_length for batch_size
                                        # = 1, otherwise, = 
                                        # train_dataset_length / batch_size
        self.test_dataset_steps = None
        self.validation_dataset_steps = None
    def use(self, *args):
        if args is not None:
            if args[0] == 'generator':
                self.use_generator(args[1])
            elif args[0] == 'direct':
                self.use_direct(args[1])
            elif args[0] == 'developer_mode' or args[0] == 'dev':
                self.use_developer_mode(*args)
            else:
                printt("Using developer mode by default", info=True)
                self.use_developer_mode(*args)

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
    def use_developer_mode(self, dataset=None):
        self.dev.on = True
        if dataset is not None:
            self.dev.dataset = dataset
    
    def skip(self, steps, current_file=None, epoch=None):
        if self.system_type.use_direct is True:
            self.Dataset.train_dataset=self.Dataset.train_dataset.skip(step)
        elif self.system_type.use_generator is True:
            self.generator_skip(steps, current_file=current_file, epoch=epoch)
    def generator_skip(self, steps, current_file, epoch):
        '''
        This function must be mainly implemented in the child class.
        It will use the function arguments to set some active variables.
        Take a look at BiobankDataLoader for an example.
        '''
        printt("Generator Skip not implemented", error=True, stop=True)

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

    def __check__(self):
        if self.train_dataset_length is None:
            printt("Dataset length not set", error=True, stop=True)

    def __call__(self, *args, **kwargs):
        self.create(*args, **kwargs)
        return self

    def create(self, threads=4):
        self.set_operation_seed()
        self.__config__()
        self.__check__()
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
        self.__process_dataset__()
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
    def __process_dataset__(self):
        '''
        For direct: Creates validation folds
        For generator: access the three generators neccesssary
        '''
        if self.config.batch_size is None:
            printt("Batch size is not set so Default is set to 1", warning=True)
            self.config.batch_size = 1
        if self.system_type.use_direct is True or self.dev.on is True:
            if self.dev.on is False:
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
                self.Datasets = [x.batch(batch_size=self.config.batch_size) for x in self.Datasets]
                self.Datasets = [x.prefetch(buffer_size=self.config.batch_size*self.config.prefetch_factor) for x in self.Datasets]
            elif self.config.dataset_split is not None:
                raise NotImplementedError('Dataset splitting not implemented yet')
        else:
            #self.Datasets = [x.batch(batch_size=self.config.batch_size) for x in self.Datasets]
            self.Datasets = [x.prefetch(buffer_size=self.config.batch_size) for x in self.Datasets]
            #self.Dataset = self.Dataset.batch(batch_size=self.config.batch_size)
            #self.Dataset = self.Dataset.prefetch(buffer_size=self.config.batch_size*self.config.prefetch_factor)
        self.set_dataset_steps()
        self.train_dataset = self.Datasets[0]
        self.test_dataset = self.Datasets[1]
        self.validation_dataset = self.Datasets[2]


    def cifar10(self):
        self.config.cv_folds = None
        self.config.dataset_split = None
        self.config.batch_size = 256
        # Load training and eval data from tf.keras
        (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.cifar10.load_data()
        printt('CIFAR10 shapes train_data, test_data, test_data, test_labels', debug=True)
        printt(train_data.shape, debug=True)
        printt(test_data.shape, debug=True)
        printt(train_labels.shape, debug=True)
        printt(test_labels.shape, debug=True)


        #train_data = train_data.reshape(-1, 32, 32, 3).astype('float32')
        train_data = np.float32(train_data) / 255.
        test_data = np.float32(test_data) / 255.

        train_labels = np.asarray(train_labels, dtype=np.int32)
        test_labels = np.asarray(test_labels, dtype=np.int32)

        #train_data = train_data[0:49920]
        #train_labels = train_labels[0:49920]

        self.train_dataset_length = train_labels.shape[0] # 50,000
        self.test_dataset_length = test_labels.shape[0] # 10,000
        self.validation_dataset_length = 1
        # for train
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
        #train_dataset_data = tf.data.Dataset.from_tensor_slices(train_data)
        #train_dataset_labels = tf.data.Dataset.from_tensor_slices(train_labels)
        #train_dataset = tf.data.Dataset.zip((train_dataset_data, train_dataset_labels))
        
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
        validation_dataset = test_dataset.take(self.validation_dataset_length)

        self.Datasets = [train_dataset, test_dataset, validation_dataset]
        printt(self.Datasets, debug=True)
        self.Datasets = [x.batch(batch_size=self.config.batch_size) for x in self.Datasets]
        self.Datasets = [x.prefetch(buffer_size=self.config.batch_size*self.config.prefetch_factor) for x in self.Datasets]


    def set_dataset_steps(self):
        if self.config.batch_size is None:
            printt("Batch size not set!", error=True, stop=True)
        if self.train_dataset_length is None:
            printt("Training dataset size not set!", error=True, stop=True)
        if self.test_dataset_length is None:
            printt("Test dataset size not set!", error=True, stop=True)
        self.train_dataset_steps = np.floor(np.divide(self.train_dataset_length,
            self.config.batch_size))
        self.test_dataset_steps = np.floor(np.divide(self.test_dataset_length,
            self.config.batch_size))

    def set_operation_seed(self, seed=1114):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        __seed__(seed)

