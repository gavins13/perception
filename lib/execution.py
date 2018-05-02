import tensorflow as tf

import sys


class execution(object):
  def __init__(self, project_path, model, data_strap, type='train', experiment_name=None):
    self.__enter__(project_path, model, data_strap, type='train', experiment_name=None)

  def __enter__(self, project_path, model, data_strap, type='train', experiment_name=None):
    # Set Saving Directories
    if(name===None):
      experiment_name = raw_input("Name of experiment: ")
    self.foldername = experiment_name + '_' + str(datetime.now())
    self.foldername_full = project_path + '/experimental_results/' + train_foldername
    os.mkdir(foldername_full, 0755  );
    print("Results will be saved to %s" % train_foldername_full)
    self.summary_folder = self.foldername_full + type

    # Set up the data
    self.data_strap = data_strap
    if(type=='train'):
        self.experiment = self.training
        self.data_strap.will_train()
    elif(type=='evaluate'):
        self.experiment = self.evaluate
        self.data_strap.will_test()
    else:
        raise Exception('experiment stage-type not valid')

    # Set up the retrieval of the results
    with tf.Graph().as_default():
    #   -->  Build model
    self.summarised_result, self.results = model.run_multi_gpu(data_strap)
    #   -->  Print stats
    self.param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.
        TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    sys.stdout.write('total_params: %d\n' % self.param_stats.total_parameters)
    self.writer = tf.summary.FileWriter(self.summary_folder)


  def run_task(self, type, task_name, DataObject):
    self.experiment(load_training, summary_dir, writer, train_experiment, result,max_steps, save_step)

  def __exit__(self):
    self.writer.close()
