import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from .misc import printt
from .customiser import CustomUserModule

class Dataset(CustomUserModule):
    '''
    The Dataset class provides the abstraction that allows:
        1. Automatic handling of training, validation and testing splitting
        2. Dataset shuffling (with automatic seed setting)
        3. Prefetching and multi-threading for Python generator functions
        4. Automatic test mode

    Usage:
    To use, simply initialise, call use(), then create().
    Afer this, self.Datasets is a list of 3 items: the training, validation and
    testing datasets that you can loop through during training/validation/etc.

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

    It is advisable to set "self.generator.num_files" to the number of training
    files, so that training display is more informative.

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

    You may also wish to preset the generator output shape. You can do this via
    the attribute
    ```
    self.generator.output_shape
    ```
    which may be a list of ints describe the output shape of the generator
    (all three generators - train, validation and test) OR it may be a list
    of lists (which the latter lists being lists of ints). The former list
    will be of size 3: train, validation and test. For example:
    ```
    self.generator.output_shape = [
        train_output_shape(or_structure),
        validation_output_shape(or_structure),
        test_output_shape(or_structure)
    ]
    ```
    This can be set in __init__ or __config__

    Developer mode:
    If using developer mode, remember to set self.dev.dataset before calling
    use()

    '''
    counter=0 # counter for threads
    def __init__(self, *args, **kwargs):
        self.gen_num = Dataset.counter
        Dataset.counter += 1

        # Define System Configuration
        class Config: pass
        self.system_type = Config()
        self.system_type.use_generator = False # Use TF Generator
        self.system_type.use_direct = False # Load all into memory

        # Define Developer Mode Settings
        self.dev = Config()
        self.dev.on = True
        self.dev.dataset = 'cifar10'

        # Define Generator Mode Settings
        self.generator = Config()
        self.generator.data_types = None
        self.generator.num_files = None
        self.generator.output_shape = None

        # Define Dataset Settings
        self.config = Config()
        self.config.type = 'train' # 'train', 'test', 'validate'
        self.config.batch_size = None # Can be int or list [train, val, test]
                                      # or dict(train,validation,test)

        # Define Advanced Dataset Settings
        self.config.epochs = None # Better to define epochs in .JSON experiment
        self.config.disable_batching = False
        self.generator.single_thread_test_dataset = True
        self.generator.single_thread = False

        # Set default data splitting values according to kwargs
        '''
        Only valid when the using direct method
        '''
        self.config.dataset_split = [60,30,10] # Train, Test, Validate
        self.config.cv_folds = None
        self.config.cv_fold_number = None
        self.config.validation_size = 1
        '''
        Select folds and fold number
        '''
        if 'cv_folds' in kwargs.keys():
            self.config.cv_folds = kwargs['cv_folds']
            if 'cv_fold_number' in kwargs.keys():
                self.config.cv_fold_number = kwargs['cv_fold_number']
            else:
                self.config.cv_fold_number = 1
            printt("Using {} folds, fold number {}".format(self.config.cv_folds, self.config.cv_fold_number))

        # Some important variables that need setting by the user
        self.train_dataset_length = None
        self.test_dataset_length = None
        self.validation_dataset_length = None
        self.operation_seed = None

        # Variables set by Perception
        self.train_dataset_steps = None
        self.test_dataset_steps = None
        self.validation_dataset_steps = None

        # Define Active Variables for this Data Loader
        self.current = Config()
        self.current.epoch = None
        self.current.step = None
        self.current.file = -1

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
        self.generator.data_types = data_types
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
        self.dataset_length = len(dataset)
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
        if (self.train_dataset_length is None) and (self.system_type.use_direct is False):
            printt("Dataset length not set", error=True, stop=True)

    def __call__(self, *args, **kwargs):
        self.create(*args, **kwargs)
        return self

    def __process_generator_shapes__(self):
        if self.generator.output_shape is None:
            self.generator.output_shape = [None, None, None]
        elif isinstance(self.generator.output_shape, int) is True:
            printt("Generator shape error. Check your property `self.generator.`.",
                error=True, stop=True)
        elif isinstance(self.generator.output_shape, list) is True:
            if isinstance(self.generator.output_shape[0], int) or isinstance(self.generator.output_shape[0], np.int):
                '''
                Either it's a list of ints or a single shape that needs to be replicated.
                Assume replication
                '''
                self.generator.output_shape = [self.generator.output_shape]*3
        else:
            self.generator.output_shape = [self.generator.output_shape]*3
        

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
                self.__process_generator_shapes__()
                these_threads = [threads]*3 if isinstance(threads, int) else threads
                test_threads = 1 if self.generator.single_thread_test_dataset is True else threads[2]
                self.Datasets = [
                    self.create_generator('py_gen_train', threads=threads[0], output_shape=self.generator.output_shape[0]),
                    self.create_generator('py_gen_validation', threads=threads[1], output_shape=self.generator.output_shape[1]),
                    self.create_generator('py_gen_test', threads=test_threads, output_shape=self.generator.output_shape[2])
                ]
                self.Dataset = self.Datasets[0]
            elif self.system_type.use_direct is True:
                # Nothing?
                printt("-")
        self.__process_dataset__()
    def create_generator(self, generator_name='py_gen_train', threads=4, output_shape=None):
        if self.generator.single_thread is False:
            dummy_ds = tf.data.Dataset.from_tensor_slices(['DataGenerator']*threads)
            dummy_ds = dummy_ds.interleave(
                lambda x: tf.data.Dataset.from_generator(
                    lambda x: getattr(self, generator_name)(x),
                    output_types=self.generator.data_types, args=(x,),
                    output_shapes=output_shape
                ),
                cycle_length=threads,
                block_length=1,
                num_parallel_calls=threads)
            return dummy_ds
        else:
            printt("Using a single-thread-mode for generator", warning=True)
            dataset = tf.data.Dataset.from_generator(
                getattr(self, generator_name),
                output_types=self.generator.data_types,
                args=(generator_name+'_0_',),
                output_shapes=output_shape
            )
            return dataset
    def __process_dataset__(self):
        '''
        For direct: Creates validation folds
        For generator: access the three generators neccesssary

        If you are replacing this method, things to be set include:
         - self.config.batch_size
         - self.Datasets : list - List of datasets in order: Train, Val, Test
         - self.train_dataset_length
         - self.test_dataset_length
         - self.validation_dataset_length
         - self.train_dataset
         - self.validation_dataset
         - self.test_dataset
        If you're not running set_dataset_steps(), then also:
        - self.train_dataset_steps
        - self.test_dataset_steps
        - self.validation_dataset_steps
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
                self.Datasets = [x.prefetch(tf.data.experimental.AUTOTUNE) if i != 2 else x for i, x in enumerate(self.Datasets) ]

                self.test_dataset_length = len([x for x in range(self.dataset_length) if (x % self.config.cv_folds) == (self.config.cv_fold_number-1)])
                self.train_dataset_length = len([x for x in range(self.dataset_length) if (x % self.config.cv_folds) != (self.config.cv_fold_number-1)])

            elif self.config.dataset_split is not None:
                raise NotImplementedError('Dataset splitting not implemented yet')
        else:
            if self.config.disable_batching is False:
                self.Datasets = [x.batch(batch_size=batch_sizes[ii]) for ii, x in enumerate(self.Datasets)]
            self.Datasets = [x.prefetch(tf.data.experimental.AUTOTUNE) if i != 2 else x for i, x in enumerate(self.Datasets) ]
        self.set_dataset_steps()
        self.train_dataset = self.Datasets[0]
        self.validation_dataset = self.Datasets[1]
        self.test_dataset = self.Datasets[2]


    def set_dataset_steps(self):
        if self.config.batch_size is None:
            printt("Batch size not set!", error=True, stop=True)
        if self.train_dataset_length is None:
            printt("Training dataset size not set!", error=True, stop=True)
        if self.test_dataset_length is None:
            printt("Test dataset size not set!", error=True, stop=True)

        batch_sizes = self.get_batch_sizes()
        self.train_dataset_steps = np.ceil(np.divide(self.train_dataset_length,
            batch_sizes[0])).astype(np.int)
        self.test_dataset_steps = np.ceil(np.divide(self.test_dataset_length,
            batch_sizes[2])).astype(np.int)
        if self.validation_dataset_length is not None:
            self.validation_dataset_steps = np.ceil(np.divide(self.validation_dataset_length,
            batch_sizes[1])).astype(np.int)

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
        '''
        Note: Other seeds are not set here anymore and are off-loaded to the
        Perception Experiment()
        '''
        self.operation_seed = seed


    @tf.function
    def rotate_and_translate(self, image, axes=[1,2], rotate=np.pi/2, translate=16):
        '''
        Data Augmentation function.
        May be moved elsewhere in future releases.

        if ndims = 3, then assume [H, W, C]
        if ndims >= 4: then assume [B, ... axes[0], ...., axes[1].... C]

        This function will apply a random 2D translation and rotation to images
        with a range specified by rotate and translate. Works with complex-
        vauled images

        Rotate can be None in which case no rotation is applied
        Translate can be None in which case no translation is applied

        Rotate can be a float which is half of the rotation range
        Rotate can be a list : [min. rotation value, max. rotation value]
        Translate can be a list of integers:
            [ half of translate_H range, half of translate_W range]
        Translate can be a list of lists of integers:
            [
                [min_translate_H value, max_translate_H value].
                [min_translate_W value, max_translate_W value].
            ]

        image can be float or complex.

        Note: H and W correspond to "Height" and "Width", usually the axes
        numbers 1 and 2 respectively.
        '''

        '''
        Construct translation and rotation values
        '''
        if isinstance(rotate, list):
            rotate_minval = rotate[0]
            rotate_maxval = rotate[1]
        elif rotate is not None:
            rotate_minval = -rotate
            rotate_maxval = rotate

        if isinstance(translate, list):
            # 1st entry is x, 2nd is y
            translate_x = translate[0]
            translate_y = translate[1]
            if isinstance(translate_x, list):
                translate_x_minval = translate_x[0]
                translate_x_maxval = translate_x[1]
            else:
                translate_x_minval = -translate_x
                translate_x_maxval = translate_x
            if isinstance(translate_y, list):
                translate_y_minval = translate_y[0]
                translate_y_maxval = translate_y[1]
            else:
                translate_y_minval = -translate_y
                translate_y_maxval = translate_y
        elif translate is not None:
            translate_x_minval = -translate
            translate_y_minval = -translate
            translate_x_maxval = translate
            translate_y_maxval = translate

        '''
        Transpose and reshape image to correct shape for TF
        '''
        original_shape = tf.shape(image)
        if (len(original_shape) > 3) and (axes is not None):
            to_transpose = list(range(len(tf.shape(image))))
            to_transpose_original = to_transpose[:]
            to_transpose[1] = axes[0]
            to_transpose[2] = axes[1]
            to_transpose[axes[0]] = 1
            to_transpose[axes[1]] = 2
            image = tf.transpose(image, to_transpose)
            after_transpose_shape = tf.shape(image)
            image = tf.reshape(image, [original_shape[0], original_shape[axes[0]], original_shape[axes[1]], -1])

        expanded = False
        if image.shape.__len__() ==3:
            image = tf.expand_dims(image, axis=0)
            expanded = True
        '''
        Construct TF translate and rotation tensors
        '''
        random_angles = tf.random.uniform(shape = (tf.shape(image)[0], ), minval = rotate_minval, maxval = rotate_maxval)

        random_x = tf.random.uniform(shape = (tf.shape(image)[0], 1), minval = translate_x_minval, maxval = translate_x_maxval)
        random_y = tf.random.uniform(shape = (tf.shape(image)[0], 1), minval = translate_y_minval, maxval = translate_y_maxval)
        translate = tf.concat([random_y, random_x], axis=1)

        '''
        Apply translation and rotation with TF
        '''
        if "complex" in str(image.dtype):
            if rotate is not None:
                real = tfa.image.rotate(tf.math.real(image),random_angles, interpolation="BILINEAR")
                imag = tfa.image.rotate(tf.math.imag(image),random_angles, interpolation="BILINEAR")
            else:
                real = tf.math.real(image)
                imag = tf.math.real(imag)
            if translate is not None:
                real = tfa.image.translate(real, translate, interpolation="NEAREST")
                imag = tfa.image.translate(imag, translate, interpolation="NEAREST")
            image = tf.complex(real, imag)
        else:
            if rotate is not None:
                image = tfa.image.rotate(image, random_angles, interpolation="BILINEAR")
            if translate is not None:
                image = tfa.image.translate(image, translate, interpolation="NEAREST")

        '''
        Transpose and reshape back to original shape
        '''
        if expanded is True:
            image = tf.squeeze(image, axis=0)

        if (len(original_shape) > 3) and (axes is not None):
            image = tf.reshape(image, after_transpose_shape)
            from_transpose = to_transpose[:]
            from_transpose[1] = axes[0]
            from_transpose[2] = axes[1]
            from_transpose[axes[0]] = 1
            from_transpose[axes[1]] = 2
            image = tf.transpose(image, from_transpose)

        return image

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
        self.Datasets = [x.prefetch(tf.data.experimental.AUTOTUNE) for x in self.Datasets]

    def reset(self):
        '''
        Reset generator counters
        '''
        self.current.file=0
        self.current.epoch=0