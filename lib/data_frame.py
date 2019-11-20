import tensorflow as tf
import numpy as np


class Data_V2(object):
    def __init__(self, data_loader, num_gpus=1, validation_size=1):
        # data_loader is a function that returns :
        # train_data, train_data_labels, test_data, test_data_labels
        self.dataset_split_size = None
        self.mini_batch_size=None
        self.validation_size=validation_size
        if(data_loader != None):
            print(">> Enter Loader")
            self.train_data, self.train_data_labels, self.test_data, self.test_data_labels, self.extra_data = data_loader() # must be np tensors
            print(">> Exit  Loader")
            self.mode = None
            self.dataset_size_train = len(self.train_data_labels)
            self.dataset_size_test = len(self.test_data_labels)
            self.set_num_gpus(num_gpus)
            self.set_validation_dataset()
        else:
            raise NotImplementedError()

    def set_num_gpus(self, num):
        ''' Sets the number of gpus that will be used and subsequently split up the data '''
        print(">>>> Let's set the number of GPUs and manage the data across them")
        self.num_gpus = num

        def splitN(N, d):
            ans = np.float(N) / np.float(d)
            Nn = np.ceil(ans)
            print(">>>>>>>>>>>>> Check divisor:  %d, %d" % (N,d))
            if(ans != np.float(np.int(ans))):
                # not a perfect divisor
                print(">>>>>>>>>>>>>>> Not a perfect divisor")
                ans = np.float(N) / np.float(d-1)
                Nn = np.ceil(ans)
                r = N - ( Nn*(d-1) )
            else:
                print(">>>>>>>>>>>>>>> Perfect divisor")
                r=Nn
            print([Nn, d,r, N])
            assert(Nn*(d-1) + r == N)
            r = None if(r==Nn) else np.int(r)
            return np.int(Nn), r

        def get_split_indices(train_length, N):
            # N is the number of splits that are required to be made
            if(N==1):
                return np.asarray([list(range(train_length))])
            else:
                train_split_size, train_final_split = splitN(train_length, N)
                train_indices = np.split(np.asarray(range((train_split_size*N))), N)
                if(train_final_split!=None):
                    np.append(train_indices, list(range(train_split_size*N,train_split_size*N +train_final_split)))
                return np.asarray(train_indices)
        self.train_indices_for_gpu = get_split_indices(self.dataset_size_train, num)
        self.test_indices_for_gpu = get_split_indices(self.dataset_size_test, num)
        self.set_mini_batch(self.mini_batch_size)
        '''
        def split_data(data, labels, d):
            if(d==1):
                return data, labels
            else:
                N_train = len(labels)
                N_train_split_size, N_train_final_split = splitN(N_train, d)
                print(">>>>>>>> Time to split data tensor across GPUs")
                print(">>>>>>>> Data first...")
                N_train_dataset_split = np.split(np.asarray(data[0:N_train_split_size*d]), d)
                print(">>>>>>>> Labels second...")
                N_train_dataset_labels_split = np.split(np.asarray(labels[0:N_train_split_size*d]), d)
                print(">>>>>>>> Makes these into a list..")
                print(np.asarray(N_train_dataset_labels_split).shape)
                N_train_dataset_split = np.asarray(N_train_dataset_split)
                N_train_dataset_labels_split = np.asarray(N_train_dataset_labels_split)
                #N_train_dataset_split = list(N_train_dataset_split)
                #N_train_dataset_labels_split = list(N_train_dataset_labels_split)
                print(">>>>>>>> Finished Making these into a list..")
                if(N_train_final_split!=None):
                    #N_train_dataset_split.append(data[N_train_split_size*d:N_train_split_size*d +N_train_final_split, :])
                    #N_train_dataset_labels_split.append(labels[N_train_split_size*d:N_train_split_size*d +N_train_final_split])
                    np.append(N_train_dataset_split, [data[N_train_split_size*d:N_train_split_size*d +N_train_final_split, :]])
                    np.append(N_train_dataset_split, [labels[N_train_split_size*d:N_train_split_size*d +N_train_final_split]])
                print(np.asarray(N_train_dataset_labels_split).shape)
                return N_train_dataset_split, N_train_dataset_labels_split
        self.train_data, self.train_data_labels = split_data(self.train_data, self.train_data_labels, self.num_gpus)
        self.test_data, self.test_data_labels = split_data(self.test_data, self.test_data_labels, self.num_gpus)
        '''
        print(">>>> Finished GPU data management")
        #self.dataset_split_size = np.size(self.train_data_a)

    def set_mini_batch(self, batch_size):
        print("Initialising Mini Batch with size %s and %s GPUs" % (batch_size, self.num_gpus))
        batch_size = 1 if(batch_size==None) else batch_size
        if(1==2):
            self.train_indices = np.asarray([[x] for x in self.train_indices_for_gpu])
            self.test_indices = np.asarray([[x] for x in self.test_indices_for_gpu])
            self.n_splits_per_gpu_train = [len(x) for x in self.train_indices_for_gpu]
            self.n_splits_per_gpu_test =  [len(x) for x in self.test_indices_for_gpu]
        else:
            if((type(batch_size) is int)==False): raise TypeError("batch_size must be an integer")
            self.n_splits_per_gpu_train = [np.int(np.floor(len(x)/np.float(batch_size))) for x in self.train_indices_for_gpu] # NB/ the use of floor() here results in the loss of some data hence it is important to pick training sizes that are multiples of 2
            self.n_splits_per_gpu_test = [np.int(np.floor(len(x)/np.float(batch_size))) for x in self.test_indices_for_gpu]
            if(self.dataset_size_train < self.num_gpus*batch_size):
                # This is the scenerio where the dataset is mainly used for testing or overfitting
                # Hence we automatically create the train dataset, ignoring the actual data present
                print("**** >>>> CHECK TRAIN DATASET - DUPLICATING.... <<<< ****")
                self.n_splits_per_gpu_train = [1 if (n_split==0) else n_split for n_split in self.n_splits_per_gpu_train]
                self.train_indices_for_gpu = [ [0]*batch_size for _ in range(self.num_gpus)]
            print(">> Perform Split")
            print(self.n_splits_per_gpu_train)
            print(batch_size)
            self.train_indices = [ np.split(np.asarray(x[0:batch_size*self.n_splits_per_gpu_train[gpu]]), self.n_splits_per_gpu_train[gpu]) for gpu,x in enumerate(self.train_indices_for_gpu)]
            self.test_indices = [ np.split(np.asarray(x[0:batch_size*self.n_splits_per_gpu_test[gpu]]), self.n_splits_per_gpu_test[gpu]) for gpu,x in enumerate(self.test_indices_for_gpu)]
            print(">> Assert 1")
            assert np.asarray(self.train_indices).shape[0] == len(self.train_indices_for_gpu)
            print(">> Assert 2")
            assert np.asarray(self.train_indices[0]).shape[1] == batch_size
        self.mini_batch_size = batch_size


    def fetch_data(self,set_type, key, gpu, batch_number):
        # set_type is one of 'train','test' and key is either 'input', 'labels' or a key from extra_data dictionary/collection
        if(batch_number==None):
            if((type(batch_number) is int)==False):
                raise TypeError('check your batch number --> fetch_data()')
        if(set_type=='train'):
            indices = self.train_indices
            if(key=='labels'):
                data = self.train_data_labels
            elif(key=='input'):
                data = self.train_data
            elif(key in self.extra_data._fields):
                data = getattr(self.extra_data, key).train
            else:
                raise KeyError('Key doesn\'t exists')
        elif(set_type=='test'):
            indices = self.test_indices
            if(key=='labels'):
                data = self.test_data_labels
            elif(key=='input'):
                data = self.test_data
            elif(key in self.extra_data._fields):
                data = getattr(self.extra_data, key).test
            else:
                raise KeyError('Key doesn\'t exists')
        elif(set_type=='validation'):
            indices = self.test_indices
            #print(indices)
            indices = [[[0]]*self.num_gpus]
            #print(">>>>>>>>>>>>>> VALIDATION INDICES")
            #print(indices)
            if(key=='labels'):
                data = self.test_data_labels
            elif(key=='input'):
                data = self.test_data
            elif(key in self.extra_data._fields):
                data = getattr(self.extra_data, key).test
            else:
                raise KeyError('Key doesn\'t exists')

            '''indices =indices[gpu][batch_number]
            indices = list(np.asarray(indices).flatten())
            data = np.asarray(data)'''
            data = np.asarray(data)
            data = data[0]
            if(len(data.shape)==2):
                data = np.expand_dims(data, axis=0)
            return data
        else:
            raise ValueError('please select test or training set')
        #indices = indices[gpu] if(batch_number==None) else indices[gpu][batch_number] # just a simple list
        indices =indices[gpu][batch_number]
        indices = list(np.asarray(indices).flatten())
        data = np.asarray(data)
        return data[indices]
        '''if(batch_number==None):
            return data[gpu] # returns [batch_number, --data-dims--...]
        else:
            return data[gpu][batch_number] # returns [--data-dims--...]'''

    def set_validation_dataset(self, validation_size=None):
        self.validation_size = validation_size if (validation_size != None) else self.validation_size
        # returns validation set on the gpu'th GPU
        if(self.validation_size==None):
            raise ValueError()
        self.validation_data =  self.test_data[0:self.validation_size]
        self.validation_data_labels = self.test_data_labels[0:self.validation_size]
        self.validation_extra_data = {}
        for key in self.extra_data._fields:
            self.validation_extra_data[key] = getattr(getattr(self.extra_data, key), 'test')[0:self.validation_size]



    def reduce_dataset(self, max_size=1):
        # max_size sets the minimum size of the mini-batch. Hence, for a dataset size = 1, you will need to self.set_num_gpus(1).
        # When you wish to overfit the dataset for testing, typically 'reduce_dataset' will be called to drastically reduce the number of training examples
        if(max_size<1):
            raise ValueError()
        self.train_data = self.train_data[0:max_size]
        self.train_data_labels = self.train_data_labels[0:max_size]
        self.dataset_size_train = max_size
        self.mode = 'train'
        self.set_num_gps(self.num_gpus)

    def split_data(self):
        raise NotImplementedError()

    def will_train(self):
        self.mode = 'train'
    def will_test(self):
        self.mode = 'test'
