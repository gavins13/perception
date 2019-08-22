impotr tensorflow as tf


class Data(object):
    def __init__(self, data_loader, num_gpus=1):
        # data_loader is a function that returns :
        # train_data, train_data_labels, test_data, test_data_labels
        if(data_loader !== None):
            self.train_data, self.train_data_labels, self.test_data, self.test_data_labels = data_loader()
            self.mode = None
            self.set_num_gpus(num_gpus)
        else:
            raise NotImplementedError()

    def set_num_gpus(self, num):
        ''' Sets the number of gpus that will be used and subsequently split up the data '''
        self.num_gpus = num
        raise NotImplementedError()

    def split_data(self):
        raise NotImplementedError()

    def will_train(self):
        self.mode = 'train'
    def will_test(self):
        self.mode = 'test'
    def get_data(self, gpu=i):
        raise NotImplemented('Needs to Multi-GPU compatibility')
        if(self.mode=='test'):
            return self.test_data, self.test_data_labels
        elif(self.mode=='train'):
            return self.train_data, self.train_data_labels
        else:
            raise Exception('invalid data mode')
