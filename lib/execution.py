import tensorflow as tf
import numpy as np
import os, sys

from datetime import datetime
import json
import time

import matplotlib
matplotlib.use("qt4agg")
import matplotlib.pyplot as plt

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
        print(">>>Set initialiser for training - i.e. set AdamOptimizer")
        self.model.initialise_training()
        print(">>>Finished setting initialiser")
        print(">> Time to build TF Graph!")
        self.summarised_result, self.results, self.ground_truths, self.input_data = self.model.run_multi_gpu(self.data_strap)
        self.saver = tf.train.Saver(max_to_keep=self.max_steps_to_save)
    print(">> Let's analyse the model parameters")
    print(">> Finished analysing")
    return self

  def run_task(self, max_steps, save_step=1, max_steps_to_save=1000):
      print(">Create TF session")

      config = tf.ConfigProto(allow_soft_placement=True)
      #config.gpu_options.allow_growth = True

      with tf.Session(graph=self.graph, config=config) as self.session:
          init_op = tf.group(tf.global_variables_initializer(),
                             tf.local_variables_initializer())
          print(">Initialise sesssion with variables")
          graph_res_fetches = self.session.run(init_op) # Initialise graph with variables
          print(">Load last saved model")
          self.last_global_step = self.load_saved_model()
          coord = tf.train.Coordinator()
          threads = tf.train.start_queue_runners(sess=self.session, coord=coord)
          try:
            self.experiment(max_steps=max_steps,save_step=save_step, session=self.session) # experiment = training() or evaluate()
          except tf.errors.OutOfRangeError:
            tf.logging.info('Finished experiment.')
          finally:
            coord.request_stop()
          coord.join(threads)
      self.session.close()
  def training(self, max_steps, save_step, session):
      step = 0
      last_epoch = int(self.last_global_step / self.data_strap.n_splits)
      last_mini_batch = self.last_global_step - (last_epoch * self.data_strap.n_splits)
      for j in range(last_epoch, max_steps):
          n_splits_list = range(last_mini_batch, self.data_strap.n_splits)
          last_mini_batch = 0
          for i in n_splits_list:
              print("training epoch: %d" % j)
              step += 1
              feed_dict = {}
              for gpu in range(self.data_strap.num_gpus):
                  train_data, train_labels = self.data_strap.get_data(gpu=gpu, mb_ind=i)
                  feed_dict["InputDataGPU" + str(gpu) + ":0"] = train_data
                  feed_dict["InputLabelsGPU" + str(gpu) + ":0"] = train_labels
              #sess.run(y, {tf.get_default_graph().get_operation_by_name('x').outputs[0]: [1, 2, 3]})
              print("data split: %d of %d" % (i, self.data_strap.n_splits))
              print("step: %d" % step)
              summary, _ = session.run([self.summarised_result.summary, self.summarised_result.train_op], feed_dict=feed_dict) # Run graph # summary_i, result, ground_truth, input_data
              self.writer.add_summary(summary, step)
              if (step + 1) % save_step == 0:
                  self.saver.save(self.session, os.path.join(self.summary_folder, 'model.ckpt'), global_step=step + 1)




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

  def evaluate(self, max_steps=None, save_step=None, session=None, checkpoint_path=None):
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
        last_step =load_model_and_last_saved_step(last_checkpoint_path)
        summaries = []
        results = []
        ground_truths = []
        input_datas = []
        for i in range(self.data_strap.n_splits):
            feed_dict = {}
            for gpu in range(self.data_strap.num_gpus):
              test_data, test_labels = self.data_strap.get_data(gpu=gpu, mb_ind=i)
              feed_dict["InputDataGPU" + str(gpu) + ":0"] = test_data
              feed_dict["InputLabelsGPU" + str(gpu) + ":0"] = test_labels
            print("data split: %d of %d" % (i, self.data_strap.n_splits))
            summary_i, result, ground_truth, input_data = self.session.run([self.summarised_result.summary, self.results, self.ground_truths, self.input_data],feed_dict=feed_dict)
            print("finished data split: %d of %d" % (i+1, self.data_strap.n_splits))


            summary_i = tf.Summary.FromString(summary_i)
            #print(summary_i)
            summary_dict = {}
            for val in summary_i.value:
                this_tag = val.tag.split('/')[-1]
                summary_dict[this_tag] = val.simple_value
            print(summary_dict)
            summaries.append(summary_dict)



            results = results + np.split(result[0], result[0].shape[0])
            ground_truths = ground_truths + np.split(ground_truth[0], result[0].shape[0])
            #print(len(np.split(input_data[0], result[0].shape[0])))
            #print(np.split(input_data[0], result[0].shape[0])[0].shape)
            input_datas = input_datas + np.split(input_data[0], result[0].shape[0])
        def _average_diagnostics(summaries):
            n = len(summaries)
            keys=list(summaries[0].keys())
            diagnostics = {}
            for key in keys:
                vals = []
                for i in range(n):
                    vals.append(summaries[i][key])
                if('min' in key):
                    diagnostics[key] = np.min(vals)
                elif('max' in key):
                    diagnostics[key] = np.max(vals)
                else:
                    diagnostics[key] = np.mean(vals)
            return diagnostics
        diagnostics = _average_diagnostics(summaries)

        path_orig = self.summary_folder + '/original/'
        os.mkdir(path_orig, 0o755 )
        path_mri = self.summary_folder + '/mri/'
        os.mkdir(path_mri, 0o755 )
        for i in range(len(results)):
            print("Saving results %s of %s" % (i, len(results)))
            print(np.squeeze(input_datas[i]).shape)
            print(np.squeeze(results[i]).shape)
            print(np.squeeze(ground_truths[i]).shape)
            fig=plt.figure()
            plt.subplot(131)
            plt.imshow(np.squeeze(input_datas[i]), cmap=plt.cm.gray)
            plt.subplot(132)
            plt.imshow(np.squeeze(results[i]), cmap=plt.cm.gray)
            plt.subplot(133)
            plt.imshow(np.squeeze(ground_truths[i]), cmap=plt.cm.gray)
            plt.savefig(path_mri + str(i) + ".png")
            plt.close(fig)


        print("Test results:")
        print(diagnostics)
        tf.logging.info(json.dumps(diagnostics))

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
