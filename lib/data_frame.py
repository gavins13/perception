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
            return np.shape(self.test_data[gpu][0:self.validation_size]), np.shape(self.test_data_labels[gpu][0:self.validation_size])
        
    def get_validation_data(self, gpu=0):
        # returns validation set on the gpu'th GPU
        if(self.validation_size==None):
            raise ValueError()
        if(self.num_gpus==1):
            return self.test_data[0:self.validation_size], self.test_data_labels[0:self.validation_siz]
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
        self.validation_size=1
        if(data_loader != None):
            self.train_data, self.train_data_labels, self.test_data, self.test_data_labels = data_loader() # must be np tensors
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
