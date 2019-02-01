import tensorflow as tf
import numpy as np
import os

from datetime import datetime
import time
import pickle


# Object is a context manager!
class execution(object):
    def __init__(self, project_path, model, data_strap, type='train', load=None, experiment_name=None, max_steps_to_save=1000, mini_batch_size=4):
        # Set Saving Directories
        if(experiment_name==None):
          experiment_name = input("Name of experiment: ")
        datetimestr = str(datetime.now())
        datetimestr = datetimestr.replace(" ", "-")

        if(load==None):
            self.foldername = experiment_name + '_' + datetimestr
            self.foldername_full = project_path + '/experimental_results/' + self.foldername
        else:
            self.foldername_full = project_path + '/experimental_results/' + load + '/'
            print("Load Dir being used.")

        print("Results will be saved to %s" % self.foldername_full)
        if load==None: os.mkdir( self.foldername_full, 0o755 )
        self.summary_folder = self.foldername_full + '/' +type + '/'
        print(">Create TF FileWriter")
        self.writer = tf.summary.FileWriter(self.summary_folder)

        self.model = model
        self.data_strap = data_strap
        self.max_steps_to_save=max_steps_to_save

        if(type=='train'):
            self.experiment = self.training
            self.data_strap.will_train()
        elif(type=="overfit"):
            self.experiment = self.training
            self.data_strap.will_train()
            self.data_strap.reduce_dataset()
        elif((type=='evaluate') or (type=='test')):
            if (load==None): raise Exception('The Model Saved directory is not set!')
            self.experiment = self.evaluate
            self.data_strap.will_test()
        else:
            raise Exception('experiment stage-type not valid')
        self.data_strap.set_mini_batch(mini_batch_size)

    def __enter__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            print(">>> Set initialiser for training - i.e. set AdamOptimizer")
            self.model.initialise_training()
            print(">>> Finished setting initialiser")

            print(">>> Build training datasets and iterators")
            self.train_data_placeholder = tf.placeholder(tf.as_dtype(self.data_strap.train_data.dtype), shape=self.data_strap.train_data.shape)
            self.train_data_labels_placeholder = tf.placeholder(tf.as_dtype(self.data_strap.train_data_labels.dtype), shape=self.data_strap.train_data_labels.shape)
            tensor_slices = [self.train_data_placeholder, self.train_data_labels_placeholder]
            self.extra_data_placeholders = {}
            print(">>>> Build placeholders")
            for key in self.data_strap.extra_data._fields:
                this_data = getattr(getattr(self.data_strap.extra_data, key), 'train')
                self.extra_data_placeholders[key] = tf.placeholder(tf.as_dtype(this_data.dtype), shape=this_data.shape)
                tensor_slices.append(self.extra_data_placeholders[key])
            self.tf_train_dataset = tf.data.Dataset.from_tensor_slices(tuple(tensor_slices))
            self.tf_train_dataset = self.tf_train_dataset.batch(self.data_strap.mini_batch_size)
            self.tf_train_dataset = self.tf_train_dataset.repeat(None) # number of epochs = None = infinity
            self.train_data_iterator = self.tf_train_dataset.make_initializable_iterator()
            train_data_gpus = []
            train_data_labels_gpus = []
            extra_data_gpus = []
            print(">>>> Build graph elements")
            for i in range(self.data_strap.num_gpus):
                graph_data = self.train_data_iterator.get_next()
                train_data = graph_data[0]
                train_data_labels = graph_data[1]
                train_data.set_shape([self.data_strap.mini_batch_size] + list(self.data_strap.train_data.shape[1::]))
                train_data_labels.set_shape([self.data_strap.mini_batch_size] + list(self.data_strap.train_data_labels.shape[1::]))
                train_data_gpus.append(train_data)
                train_data_labels_gpus.append(train_data_labels)
                extra_data = {}
                for j, key in enumerate(self.data_strap.extra_data._fields):
                    extra_data[key] = graph_data[j+2]
                    extra_data[key].set_shape([self.data_strap.mini_batch_size] + list(getattr(self.data_strap.extra_data, key).train.shape[1::]))
                extra_data_gpus.append(extra_data)


            print(">>> Build validation datasets and iterators")
            self.validation_data_placeholder = tf.placeholder(tf.as_dtype(self.data_strap.validation_data.dtype), shape=self.data_strap.validation_data.shape)
            self.validation_data_labels_placeholder = tf.placeholder(tf.as_dtype(self.data_strap.validation_data_labels.dtype), shape=self.data_strap.validation_data_labels.shape)
            tensor_slices = [self.validation_data_placeholder, self.validation_data_labels_placeholder]
            self.validation_extra_data_placeholders = {}
            print(">>>> Build placeholders")
            for key in self.data_strap.validation_extra_data.keys():
                this_data = self.data_strap.validation_extra_data[key]
                self.validation_extra_data_placeholders[key] = tf.placeholder(tf.as_dtype(this_data.dtype), shape=this_data.shape)
                tensor_slices.append(self.validation_extra_data_placeholders[key])
            self.tf_validation_dataset = tf.data.Dataset.from_tensor_slices(tuple(tensor_slices))
            self.tf_validation_dataset = self.tf_validation_dataset.batch(1)
            self.tf_validation_dataset = self.tf_validation_dataset.repeat(None) # number of epochs = None = infinity
            self.validation_data_iterator = self.tf_validation_dataset.make_initializable_iterator()
            validation_data_gpus = []
            validation_data_labels_gpus = []
            validation_extra_data_gpus = []
            validation_num_gpus = 1
            print(">>>> Build graph elements")
            for i in range(validation_num_gpus): # only need to run on 1 gpu
                graph_data = self.validation_data_iterator.get_next()
                validation_data = graph_data[0]
                validation_data_labels = graph_data[1]
                validation_data.set_shape([self.data_strap.mini_batch_size] + list(self.data_strap.validation_data.shape[1::]))
                validation_data_labels.set_shape([self.data_strap.mini_batch_size] + list(self.data_strap.validation_data_labels.shape[1::]))
                validation_data_gpus.append(validation_data)
                validation_data_labels_gpus.append(validation_data_labels)
                validation_extra_data = {}
                for j, key in enumerate(self.data_strap.validation_extra_data.keys()):
                    validation_extra_data[key] = graph_data[j+2]
                    validation_extra_data[key].set_shape([self.data_strap.mini_batch_size] + list(self.data_strap.validation_extra_data[key].shape[1::]))
                validation_extra_data_gpus.append(validation_extra_data)

            # This make the iterator a saveable object that can be reloaded when restarting training
            #saveable_train = tf.contrib.data.make_saveable_from_iterator(self.train_data_iterator)
            #tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable_train)
            '''saveable_validation = tf.contrib.data.make_saveable_from_iterator(self.validation_data_iterator)
            tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable_validation)'''

            training = {"input": train_data_gpus, "labels": train_data_labels_gpus, "extra_data": extra_data_gpus}
            validation = {"input": validation_data_gpus, "labels": validation_data_labels_gpus, "extra_data": validation_extra_data_gpus}

            print(">> Time to build TF Graph!")
            self.summarised_result, self.results, self.ground_truths, self.input_data = self.model.run_multi_gpu(self.data_strap, num_gpus=self.data_strap.num_gpus, data=training, validation_graph=False)
            self.validation_summarised_result, self.validation_results, self.validation_ground_truths, self.validation_input_data = self.model.run_multi_gpu(self.data_strap, num_gpus=validation_num_gpus, data=validation, validation_graph=True)
            self.saver = tf.train.Saver(max_to_keep=self.max_steps_to_save)
        print(">> Let's analyse the model parameters")
        print(">> Finished analysing")
        return self

    def run_task(self, max_epochs, save_step=1, max_steps_to_save=1000, memory_growth=False, validation_step=5):
          print(">Create TF session")

          config = tf.ConfigProto(allow_soft_placement=True)
          if(memory_growth is True):
              config.gpu_options.allow_growth = True

          with tf.Session(graph=self.graph, config=config) as self.session:
              train_feed_dict = { self.train_data_placeholder: self.data_strap.train_data, self.train_data_labels_placeholder: self.data_strap.train_data_labels }
              for key in self.data_strap.extra_data._fields:
                  train_feed_dict[self.extra_data_placeholders[key]] = getattr(self.data_strap.extra_data, key).train
              self.session.run(self.train_data_iterator.initializer, feed_dict=train_feed_dict)
              print(".")
              validation_feed_dict = { self.validation_data_placeholder: self.data_strap.validation_data, self.validation_data_labels_placeholder: self.data_strap.validation_data_labels }
              for key in self.data_strap.validation_extra_data.keys():
                  validation_feed_dict[self.validation_extra_data_placeholders[key]] = self.data_strap.validation_extra_data[key]
              self.session.run(self.validation_data_iterator.initializer, feed_dict=validation_feed_dict) #[0] because we want validation size = 1

              init_op = tf.group(tf.global_variables_initializer(),
                                 tf.local_variables_initializer())
              print(">Initialise sesssion with variables")
              graph_res_fetches = self.session.run(init_op) # Initialise graph with variables
              print(">Load last saved model")
              self.last_global_step = self.load_saved_model()
              coord = tf.train.Coordinator()
              threads = tf.train.start_queue_runners(sess=self.session, coord=coord)
              try:
                self.experiment(max_epochs=max_epochs,save_step=save_step, session=self.session, validation_step=validation_step) # experiment = training() or evaluate()
              except tf.errors.OutOfRangeError:
                tf.logging.info('Finished experiment.')
              finally:
                coord.request_stop()
              coord.join(threads)
          self.session.close()

    def training(self, max_epochs, save_step, session, validation_step):
          step = 0
          last_epoch = int(self.last_global_step / self.data_strap.n_splits_per_gpu_train[0]) # [] This is cheating and needs to be fixed
          last_mini_batch = self.last_global_step - (last_epoch * self.data_strap.n_splits_per_gpu_train[0]) # [] This is cheating and needs to be fixed
          step = (last_epoch*self.data_strap.n_splits_per_gpu_train[0])+last_mini_batch # [] This is cheating and needs to be fixed
          print("Saving to: cd %s; tensorboard --logdir=./ --port=6394" % self.summary_folder)
          print(last_mini_batch, last_epoch, max_epochs, self.last_global_step, self.data_strap.n_splits_per_gpu_train, self.data_strap.num_gpus)

          '''feed_dict = {}
          for gpu in list(range(self.data_strap.num_gpus)):
            for key in self.data_strap.extra_data._fields:
                feed_dict["ExtraData_"+key+"GPU"+str(gpu)+":0"] = self.data_strap.extra_data.image_data_complex.train #self.data_strap.fetch_data('train',key,gpu,i)
            for key in self.data_strap.extra_data._fields:
                feed_dict["ValidationExtraData_"+key+"GPU"+str(gpu)+":0"] = self.data_strap.extra_data.image_data_complex.train # needs fixing to a proper validation set'''


          #sess.run(y, {tf.get_default_graph().get_operation_by_name('x').outputs[0]: [1, 2, 3]})
          #next_element = iterator.get_next()
          #session.run(iterator.initializer, feed_dict=feed_dict)
          #session.run(iterator.string_handle())
          #training_handle = session.run(iterator.string_handle())

          for j in range(last_epoch, max_epochs):
              #print(".")
              n_splits_list = range(last_mini_batch, self.data_strap.n_splits_per_gpu_train[0]) # [] This is cheating and needs to be fixed
              last_mini_batch = 0
              for i in n_splits_list:
                  #print(".")
                  step += 1
                  feed_dict = {}
                  #self.data_strap.extra_data.image_data_complex.train
                  '''for gpu in list(range(self.data_strap.num_gpus)):
                      for key in self.data_strap.extra_data._fields:
                          feed_dict["ExtraData_"+key+"GPU"+str(gpu)+":0"] = self.data_strap.extra_data.image_data_complex.train #self.data_strap.fetch_data('train',key,gpu,i)
                      for key in self.data_strap.extra_data._fields:
                          feed_dict["ValidationExtraData_"+key+"GPU"+str(gpu)+":0"] = self.data_strap.extra_data.image_data_complex.train # needs fixing to a proper validation set'''
                  #sess.run(y, {tf.get_default_graph().get_operation_by_name('x').outputs[0]: [1, 2, 3]})

                  print("training epoch: %d" % j, end=";")
                  summary, _, learn_rate, diagnostics = session.run([self.summarised_result.summary, self.summarised_result.train_op, self.model._optimizer._lr, self.summarised_result.diagnostics])
                  #feed_dict=feed_dict) # Run graph # summary_i, result, ground_truth, input_data
                  print("data split: %d of %d" % (i+1, self.data_strap.n_splits_per_gpu_train[0]), end=";")# [] This is cheating and needs to be fixed

                  print("step: %d" % step, end=";")
                  print("loss: " + str(diagnostics["total_loss"]), end=";")
                  print("Learning rate: " + str(learn_rate), end='                                  \r')

                  if (step + 1) % save_step == 0:
                      self.writer.add_summary(summary, step)
                  if (step + 1) % validation_step == 0:
                      #print("save validation")
                      summary, diagnostics = session.run([self.validation_summarised_result.summary, self.validation_summarised_result.diagnostics])
                      self.writer.add_summary(summary, step)
                  if (step + 1) % save_step == 0:
                      self.saver.save(self.session, os.path.join(self.summary_folder, 'model.ckpt'), global_step=step + 1)
                  #print([last_epoch, max_epochs, len(n_splits_list)])

    def load_saved_model(self):
        def extract_step(path):
          file_name = os.path.basename(path)
          return int(file_name.split('-')[-1])

        if tf.gfile.Exists(self.summary_folder):
            ckpt = tf.train.get_checkpoint_state(self.summary_folder)
            if ckpt and ckpt.model_checkpoint_path:
              self.saver.restore(self.session, ckpt.model_checkpoint_path)
              prev_step = extract_step(ckpt.model_checkpoint_path)
            else:
              tf.gfile.DeleteRecursively(self.summary_folder)
              tf.gfile.MakeDirs(self.summary_folder)
              prev_step = 0
        else:
            tf.gfile.MakeDirs(self.summary_folder)
            prev_step = 0
        return prev_step

    def __exit__(self, exception_type, exception_value, traceback):
        print("Exectioner has been exited")

    def evaluate(self, max_steps=None, max_epochs=None, save_step=None, session=None, checkpoint_path=None, validation_step=None):
        def extract_step(path):
            file_name = os.path.basename(path)
            return int(file_name.split('-')[-1])

        def find_checkpoint(load_dir, seen_step):
            print("Search dir: %s" % load_dir)
            ckpt = tf.train.get_checkpoint_state(load_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("checkpoint path: %s" % ckpt.model_checkpoint_path)
                global_step = extract_step(ckpt.model_checkpoint_path)
                print("global step: %s" % global_step)
                if int(global_step) != seen_step:
                    return int(global_step), ckpt.model_checkpoint_path
            return -1, None

        def load_model_and_last_saved_step(ckpt_path):
            self.saver.restore(self.session, ckpt_path)
            print('model loaded successfully')
            return extract_step(ckpt_path)

        def run_evaluation(last_checkpoint_path):
            last_step = load_model_and_last_saved_step(last_checkpoint_path)
            summaries = []
            results = []
            ground_truths = []
            input_datas = []
            all_diagnostics = []
            for i in range(self.data_strap.n_splits_per_gpu_test[0]):
                # [] This is cheating and needs to be fixed
                feed_dict = {}
                for gpu in range(self.data_strap.num_gpus):
                    test_data, test_labels = self.data_strap.get_data(gpu=gpu,
                                                                      mb_ind=i)
                    validation_data, validation_labels = self.data_strap.get_validation_data(gpu=gpu)
                    feed_dict["InputDataGPU" + str(gpu) + ":0"] = test_data
                    feed_dict["InputLabelsGPU" + str(gpu) + ":0"] = test_labels
                    feed_dict["ValidationInputDataGPU" +
                              str(gpu) + ":0"] = validation_data
                    feed_dict["ValidationInputLabelsGPU" +
                              str(gpu) + ":0"] = validation_labels
                    # print(">>>>>Extra data feed_dict")
                    for key in self.data_strap.extra_data._fields:
                        feed_dict["ExtraData_"+key+"GPU"+str(gpu)
                                  + ":0"] = self.data_strap.fetch_data(
                                      'test', key, gpu, i)
                    for key in self.data_strap.extra_data._fields:
                        feed_dict["ValidationExtraData_"+key+"GPU"+str(gpu)
                                  + ":0"] = self.data_strap.fetch_data(
                                      'test', key, gpu, 0)
                        # needs fixing to a proper validation set
                print("data split: %d of %d" %
                      (i+1, self.data_strap.n_splits_per_gpu_test[0]))
                print(test_data.shape)
                print(test_labels.shape)
                print(validation_data.shape)
                print(validation_labels.shape)
                summary_i, result, ground_truth, input_data, this_split_diagnostics,this_split_full_diagnostics = self.session.run([self.summarised_result.summary, self.results, self.ground_truths, self.input_data, self.summarised_result.diagnostics, self.summarised_result.full_diagnostics],feed_dict=feed_dict)


                print("finished data split: %d of %d" % (i+1, self.data_strap.n_splits_per_gpu_test[0]))


                summary_i = tf.Summary.FromString(summary_i)
                #print(summary_i)
                summary_dict = {}
                for val in summary_i.value:
                    this_tag = val.tag.split('/')[-1]
                    summary_dict[this_tag] = val.simple_value
                print(summary_dict)
                summaries.append(summary_dict)
                print(self.data_strap.mode)

                print(">>>> testing: ")
                print(np.asarray(result).shape)
                print(np.asarray(result[0]).shape)

                def map_reduce(accumulator, pilot):
                    pilot = np.asarray(pilot)
                    s = list(pilot.shape)
                    s_1 = list(range(len(s)))
                    s_2 = s_1[2::] + [0,1]
                    pilot = np.transpose(pilot, s_2)
                    pilot = np.reshape(pilot, tuple(s[2::] + [s[0]*s[1], 1]))
                    pilot = np.squeeze(pilot, axis=len(s)-1)
                    pilot = np.transpose(pilot, [len(s)-2]+s_1[0:len(s)-2])
                    pilot = np.split(pilot, s[0]*s[1])
                    return accumulator+pilot

                print(">>>>>>>> res")
                results = map_reduce(results, result)
                print(">>>>>>>> gts")
                ground_truths =  map_reduce(ground_truths,ground_truth)
                #print(len(np.split(input_data[0], result[0].shape[0])))
                #print(np.split(input_data[0], result[0].shape[0])[0].shape)
                input_datas = map_reduce(input_datas, input_data)
                all_diagnostics.append(this_split_full_diagnostics)
            def _average_diagnostics(summaries):
                n = len(summaries)
                keys=list(summaries[0].keys())
                reduced_diagnostics = {}
                full_diagnostics = {}
                for key in keys:
                    vals = []
                    for i in range(n):
                        vals.append(summaries[i][key])
                    if('min' in key):
                        reduced_diagnostics[key] = np.min(vals)
                    elif('max' in key):
                        reduced_diagnostics[key] = np.max(vals)
                    else:
                        reduced_diagnostics[key] = np.mean(vals)
                    full_diagnostics[key] = np.array(vals).flatten()
                return reduced_diagnostics, full_diagnostics
            reduced_summaries, user_summaries = _average_diagnostics(summaries)
            reduced_diagnostics, user_diagnostics = _average_diagnostics(all_diagnostics)

            '''main_results = {"x": [], "y": [], "gt": []}
            for i in range(len(results)):  # loop over results and store in dict
                print("Saving results %s of %s" % (i, len(results)))
                print(np.squeeze(input_datas[i]).shape)
                print(np.squeeze(results[i]).shape)
                print(np.squeeze(ground_truths[i]).shape)
                main_results["x"].append(np.squeeze(input_datas[i]))
                main_results["y"].append(np.squeeze(results[i]))
                main_results["gt"].append(np.squeeze(ground_truths[i]))
            pickle.dump(main_results, open(self.summary_folder + '/main_results.p', "wb"))
            pickle.dump(user_diagnostics, open(self.summary_folder + '/user_diagnostics.p', "wb"))
            pickle.dump(user_summaries, open(self.summary_folder + '/user_summaries.p', "wb"))'''
            self.model.ArchitectureObject.analyse(user_diagnostics, reduced_diagnostics, self.summary_folder)
            print("Test results:")
            print(reduced_diagnostics)
            #tf.logging.info(json.dumps(reduced_diagnostics))

            '''summary = tf.Summary.FromString(summaries)
            summary.value.add(tag='mean_ssim', simple_value=diagnostics["mean_ssim"])

            self.writer.add_summary(summary, last_step)'''

        seen_step = -1
        paused = 0
        while paused < 360:
            print('start evaluation, model defined')
            if checkpoint_path:
                print("Checkpoint path defined")
                step = extract_step(checkpoint_path)
                last_checkpoint_path = checkpoint_path
            else:
                print("Searching for checkpoint...")
                step, last_checkpoint_path = find_checkpoint(self.foldername_full + 'train/', seen_step)
            print("Last Checkpoint: %d" % step)
            print(last_checkpoint_path)
            if step == -1:
                print("Sleeping for 5")
                time.sleep(5) # was 60 originally
                print("Finished sleeping for 5")
                paused += 1
            else:
                paused = 0
                seen_step = step
                # Run Evaluation!
                run_evaluation(last_checkpoint_path)
                if checkpoint:
                    break
