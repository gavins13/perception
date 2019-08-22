import tensorflow as tf
import numpy as np


class Data(object):
    def get_validation_data_shape(self, gpu=0):
        # returns shape of the validation set on the gpu'th GPU
        if(self.validation_size==None):
            raise ValueError()
        if(self.num_gpus==1):
            return np.shape(self.test_data[0:self.validation_size]), np.shape(self.test_data_labels[0:self.validation_size])
        elif(self.num_gpus>1):
            return (np.shape(self.test_data[gpu][0:self.validation_size]),
                   np.shape(self.test_data_labels[gpu][0:self.validation_size]))

    def get_validation_data(self, gpu=0):
        # returns validation set on the gpu'th GPU
        if(self.validation_size==None):
            raise ValueError()
        if(self.num_gpus==1):
            return self.test_data[0:self.validation_size], self.test_data_labels[0:self.validation_size]
        elif(self.num_gpus>1):
            return self.test_data[gpu][0:self.validation_size], self.test_data_labels[gpu][0:self.validation_size]

    def set_validation_dataset(self, validation_size=None):
        self.validation_size = validation_size if (validation_size != None) else self.validation_size
        if(self.validation_size<1):
            raise ValueError()
        if(self.num_gpus==1):
            self.validation_data = self.test_data[0:self.validation_size]
            self.validation_data_labels = self.test_data_labels[0:self.validation_size]
        elif(self.num_gpus>1):
            for gpu in range(self.num_gpus):
                this_size = self.test_data[gpu].shape[0]
                self.validation_size = self.validation_size if(this_size>=self.validation_size) else this_size
            for gpu in range(self.num_gpus):
                self.test_data[gpu] = self.test_data[gpu][0:self.validation_size]
                self.test_data_labels[gpu] = self.test_data_labels[gpu][0:self.validation_size]
            self.dataset_size_validation = self.validation_size * self.num_gpus

    def __init__(self, data_loader, num_gpus=1, validation_size=1):
        # data_loader is a function that returns :
        # train_data, train_data_labels, test_data, test_data_labels
        self.dataset_split_size = None
        self.mini_batch_size=None
        self.validation_size=validation_size
        if(data_loader != None):
            self.train_data, self.train_data_labels, self.test_data, self.test_data_labels, self.extra_data = data_loader() # must be np tensors
            self.mode = None
            self.set_num_gpus(num_gpus)
            self.dataset_size_train = len(self.train_data_labels)
            self.dataset_size_test = len(self.test_data_labels)
        else:
            raise NotImplementedError()

    def reduce_dataset(self, max_size=1):
        # max_size sets the minimum size of the mini-batch. Hence, for a dataset size = 1, you will need to self.set_num_gpus(1).
        if(max_size<1):
            raise ValueError()
        if(self.num_gpus==1):
            self.train_data = self.train_data[0:max_size]
            self.train_data_labels = self.train_data_labels[0:max_size]
            self.dataset_size_train = max_size

        elif(self.num_gpus>1):
            for gpu in range(self.num_gpus):
                this_size = self.train_data[gpu].shape[0]
                max_size = max_size if(this_size>=max_size) else this_size
            for gpu in range(self.num_gpus):
                self.train_data[gpu] = self.train_data[gpu][0:max_size]
                self.train_data_labels[gpu] = self.train_data_labels[gpu][0:max_size]
            self.dataset_size_train = max_size * self.num_gpus
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

        print(">>>> Finished GPU data management")
        #self.dataset_split_size = np.size(self.train_data_a)



    def split_data(self):
        raise NotImplementedError()

    def will_train(self):
        self.mode = 'train'
    def will_test(self):
        self.mode = 'test'
    def set_mini_batch(self, mb_size_approx):
        print("Initialising Mini Batch")
        self.mini_batch_size = mb_size_approx
        if(self.num_gpus>1):
            n_splits = np.ceil(self.get_size()/np.float(mb_size_approx)) # no of splits on each gpu
            self.n_splits = n_splits
            self.mini_batch_splits = []
            for gpu in range(self.num_gpus):
                if(self.mode=='train'):
                    thissize = self.train_data[gpu].shape[0]
                elif(self.mode=='test'):
                    thissize = self.test_data[gpu].shape[0]
                print(thissize)
                this_mini_batch_size = np.float(thissize)/np.float(n_splits)
                first_size = thissize - (np.floor(this_mini_batch_size)*(n_splits-1))

                first_size = np.int(first_size)
                this_mini_batch_size = np.int(this_mini_batch_size)

                thissplit_starts=[0] + list(range(first_size,thissize, this_mini_batch_size))
                thissplit_ends = [first_size] + list(range(first_size+this_mini_batch_size, thissize+this_mini_batch_size, this_mini_batch_size))
                thissplit = list(zip(thissplit_starts, thissplit_ends))
                assert len(thissplit) == n_splits
                self.mini_batch_splits.append(thissplit)
        else:
            this_starts = list(range(0,self.get_size(), self.mini_batch_size))
            this_ends = list(range(self.mini_batch_size,self.get_size()+self.mini_batch_size, self.mini_batch_size))
            self.mini_batch_splits = list(zip(this_starts, this_ends))
            self.n_splits = len(self.mini_batch_splits)
        print(self.mini_batch_size)
        print(self.mini_batch_splits)
        print(self.n_splits)
    def get_data(self, gpu=0,mb_ind=None):
        if(self.mini_batch_size!=None):
            if(mb_ind>self.n_splits):
                raise OutOfBoundsError()
            if(self.num_gpus==1):
                start,end = self.mini_batch_splits[mb_ind]
                if(self.mode=='test'):
                    return self.test_data[start:end], self.test_data_labels[start:end]
                elif(self.mode=='train'):
                    return self.train_data[start:end], self.train_data_labels[start:end]
                else:
                    raise Exception('invalid data mode')
            elif(self.num_gpus>1):
                start,end = self.mini_batch_splits[gpu][mb_ind]
                if(self.mode=='test'):
                    return self.test_data[gpu][start:end], self.test_data_labels[gpu][start:end]
                elif(self.mode=='train'):
                    return self.train_data[gpu][start:end], self.train_data_labels[gpu][start:end]
                else:
                    raise Exception('invalid data mode')
        else:
            if(self.num_gpus==1):
                if(self.mode=='test'):
                    return self.test_data, self.test_data_labels
                elif(self.mode=='train'):
                    return self.train_data, self.train_data_labels
                else:
                    raise Exception('invalid data mode')
            elif(self.num_gpus>1):
                if(self.mode=='test'):
                    return self.test_data[gpu], self.test_data_labels[gpu]
                elif(self.mode=='train'):
                    return self.train_data[gpu], self.train_data_labels[gpu]
                else:
                    raise Exception('invalid data mode')
    def get_data_shape(self, gpu=0, mb_ind=0):
        if(self.mini_batch_size!=None):
            if(mb_ind>self.n_splits):
                raise OutOfBoundsError()
            if(self.num_gpus==1):
                start,end = self.mini_batch_splits[mb_ind]
                if(self.mode=='test'):
                    return np.shape(self.test_data[start:end]), np.shape(self.test_data_labels[start:end])
                elif(self.mode=='train'):
                    return np.shape(self.train_data[start:end]), np.shape(self.train_data_labels[start:end])
                else:
                    raise Exception('invalid data mode')
            elif(self.num_gpus>1):
                start,end = self.mini_batch_splits[gpu][mb_ind]
                if(self.mode=='test'):
                    return np.shape(self.test_data[gpu][start:end]), np.shape(self.test_data_labels[gpu][start:end])
                elif(self.mode=='train'):
                    return np.shape(self.train_data[gpu][start:end]), np.shape(self.train_data_labels[gpu][start:end])
                else:
                    raise Exception('invalid data mode')
        else:
            if(self.num_gpus==1):
                if(self.mode=='test'):
                    return np.shape(self.test_data), np.shape(self.test_data_labels)
                elif(self.mode=='train'):
                    return np.shape(self.train_data), np.shape(self.train_data_labels)
                else:
                    raise Exception('invalid data mode')
            elif(self.num_gpus>1):
                if(self.mode=='test'):
                    return np.shape(self.test_data[gpu]), np.shape(self.test_data_labels[gpu])
                elif(self.mode=='train'):
                    return np.shape(self.train_data[gpu]), np.shape(self.train_data_labels[gpu])
                else:
                    raise Exception('invalid data mode')

    def get_size(self):
        if(self.mode=='train'):
            return self.dataset_size_train
        elif(self.mode=='test'):
            return self.dataset_size_test
        else:
            raise Exception('unknown size of dataset for this mode')




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
    def get_validation_data_shape(self, gpu=0):
        # returns shape of the validation set on the gpu'th GPU
        if(self.validation_size==None):
            raise ValueError()
        #if(self.num_gpus==1):
        #    return [len(self.train_indices[0])] + np.shape(self.train_data[0]), [len(self.train_indices[0])] + np.shape(self.train_data_labels[0])
        #elif(self.num_gpus>1):
        return [len(self.train_indices[gpu][0])] + list(np.asarray(self.train_data[0]).shape), [len(self.train_indices[gpu][0])] + list(np.asarray(self.train_data_labels[0]).shape)

    def get_extra_data_shapes(self, gpu=0):
        extra_data_shapes =  {}
        print(">> Extra data shape searching....")
        for key in list(self.extra_data._fields):
            print(">>>> Key loop")
            print(key)
            if(self.num_gpus==1):
                extra_data_shapes[key] =  [len(self.train_indices[0][0])] + list(np.asarray(getattr(self.extra_data, key)[0]).shape)[1::]
            elif(self.num_gpus>1):
                extra_data_shapes[key] =  [len(self.train_indices[gpu][0])] + list(np.asarray(getattr(self.extra_data, key)[0]).shape)[1::]  ## [] redundant code here.
        print(">> Extra data shape found")
        print(extra_data_shapes)
        return extra_data_shapes
    def get_validation_extra_data_shapes(self, gpu=0):
        extra_data_shapes =  {}
        print(">> Extra data shape searching....")
        for key in list(self.extra_data._fields):
            print(">>>> Key loop")
            print(key)
            if(self.num_gpus==1):
                extra_data_shapes[key] =  [len(self.test_indices[0][0])] + list(np.asarray(getattr(self.extra_data, key)[1]).shape)[1::]
            elif(self.num_gpus>1):
                extra_data_shapes[key] =  [len(self.test_indices[gpu][0])] + list(np.asarray(getattr(self.extra_data, key)[1]).shape)[1::]  ## [] redundant code here.
        print(">> Extra data shape found")
        print(extra_data_shapes)
        return extra_data_shapes

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
    def get_validation_data(self, gpu=0):
        # returns validation set on the gpu'th GPU
        if(self.validation_size==None):
            raise ValueError()
        data=self.fetch_data('test','input', gpu,0)
        labels=self.fetch_data('test','labels', gpu,0)
        return data[0:self.validation_size], labels[0:self.validation_size]
        #return self.fetch_data('test','input',gpu, mb_ind),self.fetch_data('test','labels',gpu, 0)


    def set_validation_dataset(self, validation_size=None):
        self.validation_size = validation_size if (validation_size != None) else self.validation_size
        if(self.validation_size<1):
            raise ValueError()
        self.validation_size = validation_size if(self.mini_batch_size>=validation_size) else self.mini_batch_size

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
    def get_data(self, gpu=0,mb_ind=None): # For use with just the trianing and test date (NOT THE EXTRA_DATA)
        if(self.mini_batch_size!=None):
            #if(mb_ind>self.n_splits):
            #   raise OutOfBoundsError() [] Needs correcting!
            return self.fetch_data(self.mode,'input',gpu, mb_ind),self.fetch_data(self.mode,'labels',gpu, mb_ind)
        else:
            raise NotImplementedError()
            return self.fetch_data(self.mode,'input',gpu, 0),self.fetch_data(self.mode,'labels',gpu, 0)

    def get_data_shape(self, gpu=0, mb_ind=0):
        if(self.mini_batch_size!=None):
            if(  ( (self.mode=='train') and (mb_ind+1>self.n_splits_per_gpu_train[gpu]))    or    ((self.mode=='test') and (mb_ind+1>self.n_splits_per_gpu_test[gpu]))  ):
                raise OutOfBoundsError()
            if(self.mode=='train'):
                return [self.mini_batch_size]  + list(np.shape(self.train_data[0])),[self.mini_batch_size]  + list(np.shape(self.train_data_labels[0]))
            elif(self.mode=='test'):
                return [self.mini_batch_size]  + list(np.shape(self.test_data[0])),[self.mini_batch_size]  + list(np.shape(self.test_data_labels[0]))
        else:
            if(self.mode=='train'):
                return [len(test_indices_for_gpu[gpu])]  + list(np.shape(self.test_data[0])),[len(test_indices_for_gpu[gpu])]  + list(np.shape(self.test_data_labels[0]))
            elif(self.mode=='test'):
                return [len(train_indices_for_gpu[gpu])]  + list(np.shape(self.train_data[0])),[len(train_indices_for_gpu[gpu])]  + list(np.shape(self.train_data_labels[0]))

    def get_size(self):
        if(self.mode=='train'):
            return self.dataset_size_train
        elif(self.mode=='test'):
            return self.dataset_size_test
        else:
            raise Exception('unknown size of dataset for this mode')
