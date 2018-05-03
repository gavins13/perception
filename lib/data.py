import tensorflow as tf
import numpy as np

class Data(object):
    def __init__(self, data_loader, num_gpus=1):
        # data_loader is a function that returns :
        # train_data, train_data_labels, test_data, test_data_labels
        self.dataset_split_size = None
        if(data_loader !== None):
            self.train_data, self.train_data_labels, self.test_data, self.test_data_labels = data_loader() # must be np tensors
            self.mode = None
            self.set_num_gpus(num_gpus)
            self.dataset_size_train = np.size(self.train_data_labels)
            self.dataset_size_test = np.size(self.test_data_labels)
        else:
            raise NotImplementedError()

    def set_num_gpus(self, num):
        ''' Sets the number of gpus that will be used and subsequently split up the data '''
        self.num_gpus = num

        def splitN(N, d):
            ans = float(N) / float(d)
            Nn = np.ceil(ans)
            if(ans === float(int(ans))):
                # not a perfect divisor
                r = N - ( Nn*(d-1) )
            else:
                r=None
            assert(Nn*(d-1) + r == N)
            return Nn, r

        def split_data(data, labels, d):
            N_train = np.size(labels)
            N_train_split_size, N_train_final_split = splitN(N_train, d)
            N_train_dataset_split = tf.split(data[0:N_train_split_size*d, :], d)
            N_train_dataset_labels_split = tf.split(labels[0:N_train_split_size*d, :], d)
            N_train_dataset_split = list(N_train_dataset_split)
            N_train_dataset_labels_split = list(N_train_dataset_labels_split)

            if(N_train_final_split!==None):
                N_train_dataset_split.append(data[N_train_split_size*d:N_train_split_size*d +N_train_final_split, :])
                N_train_dataset_labels_split.append(labels[N_train_split_size*d:N_train_split_size*d +N_train_final_split, :])
            return N_train_dataset_split, N_train_dataset_labels_split

        self.train_data, self.train_data_labels = split_data(self.train_data, self.train_data_labels, self.num_gpus)

        self.test_data, self.test_data_labels = split_data(self.test_data, self.test_data_labels, self.num_gpus)

        #self.dataset_split_size = np.size(self.train_data_a)



    def split_data(self):
        raise NotImplementedError()

    def will_train(self):
        self.mode = 'train'
    def will_test(self):
        self.mode = 'test'
    def get_data(self, gpu=0):
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

    def get_size(self):
        if(self.mode=='train'):
            return self.dataset_size_train
        elif(self.mode=='test'):
            return self.dataset_size_test
        else:
            raise Exception('unknown size of dataset for this mode')
