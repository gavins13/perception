import numpy as np
#from random import shuffle, seed as __seed__
import tensorflow as tf
from .misc import printt
from .customiser import CustomUserModule

class Dataset(CustomUserModule):
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

    In __config__ (or __init__), remember to set:
     - self.train_dataset_length (along with validation and test)
     - self.train_dataset_steps
     - self.config.batch_size (int or list [train,val,test])

    NOTE: the variable self.current.steps is not used in the main Executor
    model and is purely for the Dataset API (e.g. for restarting training,
    See the skip() method for this class).

    NOTE: cv_fold_number varies from 1 to cv_folds (index doesn't start at 0!).

    Generator mode:
    When the generator mode is used, and the generator is specified, remember
    to specify in the inherited 'generator' class a '__config__' method which
    will split the generator sources (e.g. files) into 'self.config.folds'
    number of folds, and then also select a fold that the class' Python
    generators can use.

    It is advisable to set "self.num_files" to the number of training files, so
    that training display is more informative.

    Remember to define self.train_dataset_length, self.test_dataset_length
    and self.validation_dataset_length in the class

    Also, you must define a generator_skip() method. The purpose of this
    is the ensure that when training is stopped and restarted, the generator
    starts from the current location in the dataset. The code for this method
    will probably look something like this:

    ```
    def generator_skip(self, steps, current_file, epoch):
        self.current.step = steps
        self.current.file = current_file
        self.current.epoch = epoch
    ```

    Since the generator can be batch using the TF.Dataset batch() method, it is
    uneccessary in most cases to batch within the generator function. However,
    there will exist cases where you do want to batch within the generator (for
    example, when each image generated has a different image dimension). In
    this scenerio, you can disabled the TF.Dataset batch() call by using
    self.config.disable_batching = True.

    Developer mode:
    If using developer mode, remember to set self.dev.dataset before calling
    use()

    '''
    counter=0 # counter for threads
    def __init__(self, *args, **kwargs):
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
        self.config.batch_size = None # Can be int or list [train, val, test]
                                      # or dict(train,validation,test)
        self.config.prefetch_factor = 4
        self.config.epochs = None
        self.config.disable_batching = False
        self.generator.single_thread_test_dataset = True

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
        self.operation_seed = None

        if 'threads' in kwargs.keys():
            printt("Threads specified. USING %d THREADS" % kwargs['threads'], info=True)
            self.config.threads = kwargs['threads']
        else:
            printt("Threads not specified. USING 4 THREADS", info=True)
            self.config.threads = 4
        self.__kwargs__ = kwargs # Required for adding module arguments in JSON

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
        # NB/ Something like this should work:
        '''
        inputs = dict(images=images)
        outputs = dict(labels=labels)
        dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
        '''
        self.Dataset = tf.data.Dataset.from_tensor_slices((dataset))
        #raise NotImplementedError()
    def use_developer_mode(self, dataset=None):
        self.dev.on = True
        if dataset is not None:
            self.dev.dataset = dataset

    def skip(self, steps, current_file=None, epoch=None):
        if self.system_type.use_direct is True:
            steps_to_restore = int(steps % self.train_dataset_steps)
            #print(steps_to_restore, steps, self.train_dataset_steps, 'test')
            self.train_dataset=self.train_dataset.skip(steps_to_restore)
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
        '''
        How to use the generators in Perception 2.0.

        Typically three functions will need to be defined in the inherited
        class. They are:
            - py_gen_train
            - py_gen_validation
            - py_gen_test

        For each of these three types of execution of the model, there will
        be a unique instantiation of the Perception Dataset class.

        Each 'py_gen_*' function will typically involve 3 variables required by
        the Perception Dataset object:
             - self.current.epoch: (required) initialised with -1
             - self.current.step: (required) initialised with 0 (see below)
             - self.current.file: (required: when training is resumed, it is
                                    important know which file to start with)
                                   Inialised with -1
        NOTE: these variables are all for training (not validation or testing).
        Using these variables, the generator should move through a set of files
        and 'yield' an output whilst incrementing the variables above whilst
        being sure to reset self.current.file when the end of the file list has
        been reached.

        Please note, you may require counters for the validation and testing,
        but they do not need to be an object property (since the Executor does
        not need to access them).

        If the generator is being used for the first time (e.g. training for
        the first time), then self.current.step is None which is why the generator
        code will typically contain the following lines:

        ```
        if self.current.step is None:
            self.current.epoch = -1
            self.current.step = 0
            self.current.file = -1
        ```
        '''
        gen_name = gen_name.decode('utf8') + '_' + str(self.gen_num)
        epochs = -999 if self.config.epochs is None else self.config.epochs
        if self.current.step is None:
            self.current.epoch = -1
            self.current.step = 0
            self.current.file = -1
        while self.current.epoch != epochs+1:
            #sleep(0.3)
            self.current.file += 1
            this_file = [
                            load_file_data(filenames[self.current.file]) \
                            for x in range(self.config.batch_size)
                        ]
            #yield '{} yields {}'.format(gen_name, num)
            yield tf.concat(this_file, axis=0)
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

    def create(self, threads=None):
        threads = threads if threads is not None else self.config.threads
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
                test_threads = 1 if self.generator.single_thread_test_dataset is True else threads
                self.Datasets = [
                    self.create_generator('py_gen_train', threads=threads),
                    self.create_generator('py_gen_validation', threads=threads),
                    self.create_generator('py_gen_test', threads=test_threads)
                ]
                self.Dataset = self.Datasets[0]
            elif self.system_type.use_direct is True:
                # Nothing?
                printt("-")
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

        If you are replacing this method, things to be set include:
         - self.config.batch_size
         - self.Datasets : list - List of datasets in order: Train, Val, Test
         - self.train_dataset_length
         - self.test_dataset_length
         - self.train_dataset
         - self.validation_dataset
         - self.test_dataset
        If you're not running set_dataset_steps(), then also:
        - self.train_dataset_steps
        - self.test_dataset_steps
        '''
        if self.config.batch_size is None:
            printt("Batch size is not set so Default is set to 1", warning=True)
            self.config.batch_size = 1

        batch_sizes = self.get_batch_sizes()

        if self.system_type.use_direct is True or self.dev.on is True:
            if self.dev.on is False:
                buffer_size = None if not(hasattr(self, 'buffer_size')) else getattr(self, 'buffer_size')
                self.Dataset = self.Dataset.shuffle(
                    buffer_size=buffer_size, seed=self.operation_seed) if\
                    buffer_size is not None else self.Dataset
            if self.config.cv_folds is not None:
                self.Datasets = []
                if self.config.cv_fold_number is None:
                    raise ValueError('Invalid fold number specified: ', self.config.cv_fold_number)
                train_ds = []
                for i in range(self.config.cv_folds):
                    tmp = self.Dataset.shard(self.config.cv_folds, i)
                    if self.config.cv_fold_number == i+1:
                        test_ds = tmp
                    else:
                        train_ds.append(tmp)
                first = train_ds[0]
                del(train_ds[0])
                for x in train_ds:
                    first = first.concatenate(x)
                train_ds = first
                validation_ds = test_ds.take(self.config.validation_size)
                self.Datasets = [train_ds, validation_ds, test_ds]
                if self.config.disable_batching is False:
                    self.Datasets = [x.batch(batch_size=batch_sizes[ii]) for ii, x in enumerate(self.Datasets)]
                #self.Datasets = [x.prefetch(buffer_size=self.config.batch_size*self.config.prefetch_factor) for x in self.Datasets]
                self.Datasets = [x.prefetch(tf.data.experimental.AUTOTUNE) if i != 2 else x for i, x in enumerate(self.Datasets) ]
            elif self.config.dataset_split is not None:
                raise NotImplementedError('Dataset splitting not implemented yet')
        else:
            if self.config.disable_batching is False:
                #self.Datasets = [x.batch(batch_size=self.config.batch_size) for x in self.Datasets]
                self.Datasets = [x.batch(batch_size=batch_sizes[ii]) for ii, x in enumerate(self.Datasets)]
            #self.Datasets = [x.prefetch(buffer_size=self.config.batch_size*self.config.prefetch_factor) for x in self.Datasets]
            self.Datasets = [x.prefetch(tf.data.experimental.AUTOTUNE) if i != 2 else x for i, x in enumerate(self.Datasets) ]
            #self.Dataset = self.Dataset.batch(batch_size=self.config.batch_size)
            #self.Dataset = self.Dataset.prefetch(buffer_size=self.config.batch_size*self.config.prefetch_factor)
        self.set_dataset_steps()
        self.train_dataset = self.Datasets[0]
        self.validation_dataset = self.Datasets[1]
        self.test_dataset = self.Datasets[2]


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

        batch_sizes = self.get_batch_sizes()
        self.train_dataset_steps = np.floor(np.divide(self.train_dataset_length,
            batch_sizes[0]))
        self.test_dataset_steps = np.floor(np.divide(self.test_dataset_length,
            batch_sizes[2]))

    def get_batch_sizes(self):
        '''
        Interprets the user specific self.config.batch_size variable
        and then return a list with three elements:
        [train_batch_size, validation_batch_size, test_batch_size]
        '''
        if isinstance(self.config.batch_size, list):
            batch_sizes = self.config.batch_size
        elif isinstance(self.config.batch_size, dict):
            batch_sizes = [
                self.config.batch_size["train"],
                self.config.batch_size["validation"],
                self.config.batch_size["test"]
            ]
        else:
            batch_sizes = [self.config.batch_size]*3
        return batch_sizes

    def set_operation_seed(self, seed=1114):
        #tf.random.set_seed(seed)
        #np.random.seed(seed)
        #__seed__(seed)
        self.operation_seed = seed
