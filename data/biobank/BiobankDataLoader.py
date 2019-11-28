impotr json
import sys
sys.path.insert(0, '../lib/data_frame.py')
from data_frame import Dataset
import numpy as np
import nibabel

biobank_list_path = 'biobank.json'

class BioBank(Dataset):
    def import_list(self):
        self.file_list = json.load(biobank_list_path)
    def set_operation_seed(self, seed=1114):
        np.random.seed(seed)
    def shuffle_and_split(self, cv_folds=3, cv_fold_num=1):
        self.file_list = np.random.shuffle(self.file_list)
        n = len(self.file_list)
        n_ = np.int(np.float(n)/np.foat(cv_folds))
        fold_remainder = n-((cv_folds-1)*n_)
        fold_sizes = [n_]*(cv_folds-1) + [fold_remainder]
        self.train_file_list = self.file_list[0:fold_sizes[0]]
        self.test_file_list = self.file_list[fold_sizes[0]:fold_sizes[0]+fold_sizes[1]]]
        self.validation_file_list = [self.test_file_list[0]]

    def config(self):
        '''
        This is executed when create() is called. It is the first method
        executed in create()
        '''
        if self.config.batch_size != 1:
            #raise ValueError('This BioBank dataset only' + \
            # ' supports batch size 1 due to images being different sizes')
            printt("Note: batching along the slice axis", warning=True)

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
            if current_num_slices >= current_batch_max:
                # LOAD NEXT FILE
                self.current.file += 1
                idx = self.current.file % len(self.train_file_list)
                filename = self.train_file_list[self.current.file]
                data = nibabel.load(filename)
                d = data.get_fdata() # numpy data # [H, W, SLICE, TIME] # (210, 208, 11, 50) 
                current_batch_max = d.shape[2]
                d = np.transpose(d, [0,1,3,2]) # [H, W, TIME, SLICE]
                current_batch_size = self.config.batch_size if \
                  self.config.batch_size < batch_max else batch_max
            #else:
            #    # CURRENT FILE
            #    pass

            this_batch_size = current_batch_size if \
              (current_batch_max-current_num_slices) > \ 
              current_batch_size else \
              (current_batch_max-current_num_slices)
            this_data = d[:,:,:,no_slices:no_slices+this_batch_size]
            no_slices += this_batch_size
            self.current.step += 1
            if idx == 0:
                self.current.epoch += 1
            yield this_data


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
                current_batch_size = self.config.batch_size if \
                  self.config.batch_size < batch_max else batch_max
            #else:
            #    # CURRENT FILE
            #    pass

            this_batch_size = current_batch_size if \
              (current_batch_max-current_num_slices) > \ 
              current_batch_size else \
              (current_batch_max-current_num_slices)
            this_data = d[:,:,:,no_slices:no_slices+this_batch_size]
            no_slices += this_batch_size
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
                current_batch_size = self.config.batch_size if \
                  self.config.batch_size < batch_max else batch_max
            #else:
            #    # CURRENT FILE
            #    pass

            this_batch_size = current_batch_size if \
              (current_batch_max-current_num_slices) > \ 
              current_batch_size else \
              (current_batch_max-current_num_slices)
            this_data = d[:,:,:,no_slices:no_slices+this_batch_size]
            no_slices += this_batch_size
            current_step += 1
            if idx == 0:
                current_epoch += 1
            yield this_data