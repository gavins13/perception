import tensorflow as tf
import numpy as np
import os, sys

from datetime import datetime

# Object is a context manager!
class execution(object):
  def __init__(self, project_path, model, data_strap, type='train', load=None, experiment_name=None, max_steps_to_save=1000):
    # Set Saving Directories
    if(experiment_name==None):
      experiment_name = input("Name of experiment: ")
    datetimestr = str(datetime.now())
    datetimestr = datetimestr.replace(" ", "-")
    self.foldername = experiment_name + '_' + datetimestr
    self.foldername_full = project_path + '/experimental_results/' + self.foldername
    print("Results will be saved to %s" % self.foldername_full)
    os.mkdir( self.foldername_full, 0o755 );
    self.summary_folder = self.foldername_full + '/' +type + '/'

    print(">Create TF FileWriter")
    #self.writer = tf.summary.FileWriter(self.summary_folder)
    self.writer = tf.contrib.summary.create_file_writer(self.summary_folder)
    self.writer.set_as_default()
    tf.contrib.summary.always_record_summaries()

    self.model = model
    # Set up the data
    self.data_strap = data_strap
    self.max_steps_to_save=max_steps_to_save
    if(type=='train'):
        self.experiment = self.training
        self.data_strap.will_train()
    elif(type=='evaluate'):
        self.experiment = self.evaluate
        self.data_strap.will_test()
        if(load==None):
            raise Exception('The Model Saved directory is not set!')
        self.load_folder = self.foldername_full + '/' + load + '/'
    else:
        raise Exception('experiment stage-type not valid')
  def __enter__(self):
    # Set up the retrieval of the results (I.e. Build the Summarisd results, and results Graph)
    self.graph = tf.Graph() # [] [unfinished]
    with self.graph.as_default():
        #   -->  Build model
        print(">>>Set initialiser for training - i.e. set AdamOptimizer")
        self.model.initialise_training()
        print(">>>Finished setting initialiser")
        print(">> Time to build TF Graph!")
        self.summarised_result, self.results = self.model.run_multi_gpu(self.data_strap)
        #print(">>>Set training operation")
        #self.train_op = self.model._optimizer.minimize(self.summarised_result.total_loss)
        self.saver = tf.train.Saver(max_to_keep=self.max_steps_to_save)
    #   -->  Print stats
    print(">> Let's analyse the model parameters")
    #self.param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
    #    tf.get_default_graph(),
    #    tfprof_options=tf.contrib.tfprof.model_analyzer.
    #    TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    #[] [unfinished]
    #opts = (tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    #self.param_stats = tf.profiler.profile(tf.get_default_graph(),options=opts, cmd='scope')
    #print(">>>Number:")
    #sys.stdout.write('total_params: %d\n' % self.param_stats.total_parameters)
    #print(self.param_stats)
    '''total_parameters = 0
    for variable in tf.trainable_variables():
      # shape is an array of tf.Dimension
      shape = variable.get_shape()
      #print(shape)
      #print(len(shape))
      variable_parameters = 1
      for dim in shape:
        #print(dim)
        variable_parameters *= dim.value
      #print(variable_parameters)
      total_parameters += variable_parameters
    print(total_parameters)'''
    print(">> Finished analysing")
  #def __enter__(self):
  #    print(">Create TF FileWriter")
  #    #self.writer = tf.summary.FileWriter(self.summary_folder)
  #    self.writer = tf.contrib.summary.create_file_writer(self.summary_folder)
  #    self.writer.set_as_default()
  #    self.contrib.summary.always_record_summaries()

  def run_task(self, max_steps, save_step=1, max_steps_to_save=1000):
      # save_step defines the increment amount before saving a new checkpoint
      print(">Create TF session")
      #print(self.graph)
      with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True)) as self.session:
          init_op = tf.group(tf.global_variables_initializer(),
                             tf.local_variables_initializer())
          print(">Initialise sesssion with variables")
          graph_res_fetches = self.session.run(init_op) # Initialise graph with variables

          '''for i in range(max_steps):
            print("Train Iteration #%d" % i)
            self.model._global_step +=1
            graph_res_fetches = self.session.run(self.train_op)
            if(self.model._global_step>max_steps):
              raise tf.errors.OutOfRangeError()
          #[]print(">Create TF Saver")'''
          #[]self.saver = tf.train.Saver(max_to_keep=max_steps_to_save)
          print(">Load last saved model")
          self.last_step = self.load_saved_model()
          coord = tf.train.Coordinator()
          threads = tf.train.start_queue_runners(sess=self.session, coord=coord)
          try:
            self.training(max_steps=max_steps,save_step=save_step)
          except tf.errors.OutOfRangeError:
            tf.logging.info('Finished experiment.')
          finally:
            coord.request_stop()
          coord.join(threads)
      self.session.close()
  def training(self, max_steps, save_step):
    step = 0
    for i in range(self.last_step, max_steps):
        print("training: %d" % i)
        step += 1
        summary, _ = self.session.run([self.summarised_result.summary, self.summarised_result.train_op]) # Run graph
        print("training: %d" % i)
        self.writer.add_summary(summary, i)
        print("training: %d" % i)
        if (i + 1) % save_step == 0:
            print("training: %d" % i)
            self.saver.save(self.session, os.path.join(self.summary_folder, 'model.ckpt'), global_step=i + 1)



  def load_saved_model(self):
    """Loads a saved model into current session or initializes the directory.

    If there is no functioning saved model or FLAGS.restart is set, cleans the
    load_dir directory. Otherwise, loads the latest saved checkpoint in load_dir
    to session.

    Args:
    saver: An instance of tf.train.saver to load the model in to the session.
    session: An instance of tf.Session with the built-in model graph.
    load_dir: The directory which is used to load the latest checkpoint.

    Returns:
    The latest saved step.
    """
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
    print("Exectioner exitted")
    #self.writer.close() # not valid for eager execution
  # Task - Training

  # Task - Evaluation

  def evaluate(self, max_steps=None, save_step=None, checkpoint_path=None):

    def extract_step(path):
      file_name = os.path.basename(path)
      return int(file_name.split('-')[-1])
    def find_checkpoint(load_dir, seen_step):
      ckpt = tf.train.get_checkpoint_state(load_dir)
      if ckpt and ckpt.model_checkpoint_path:
        global_step = extract_step(ckpt.model_checkpoint_path)
        if int(global_step) != seen_step:
          return int(global_step), ckpt.model_checkpoint_path
      return -1, None
    def load_model_and_last_saved_step(self, load_dir):
        """Loads the latest saved model to the given session.

        Args:
        saver: An instance of tf.train.saver to load the model in to the session.
        session: An instance of tf.Session with the built-in model graph.
        load_dir: The path to the latest checkpoint.

        Returns:
        The latest saved step.
        """
        self.saver.restore(self.session, load_dir)
        print('model loaded successfully')
        return extract_step(load_dir)

    def run_evaluation(self, last_checkpoint_path):
        last_step =load_model_and_last_saved_step(self, last_checkpoint_path)
        total_correct = 0
        total_almost = 0
        max_steps = self.data_strap.get_size()
        for _ in range(max_steps):
            summary_i, correct, almost = self.session.run([self.result.summary, self.result.correct, self.result.almost]) # fetch the results output defined in multi_gpu_frame 'graph'
        total_correct += correct
        total_almost += almost

        #total_false = max_steps * 100 - total_correct # why * 100?
        #total_almost_false = max_steps * 100 - total_almost

        total_false = max_steps - total_correct # why * 100?
        total_almost_false = max_steps - total_almost

        summary = tf.Summary.FromString(summary_i)
        summary.value.add(tag='correct_prediction', simple_value=total_correct)
        summary.value.add(tag='wrong_prediction', simple_value=total_false)
        summary.value.add(
          tag='almost_wrong_prediction', simple_value=total_almost_false)
        print('Total wrong predictions: {}, wrong percent: {}%'.format(
          total_false, total_false / max_steps))
        tf.logging.info('Total wrong predictions: {}, wrong percent: {}%'.format(
          total_false, total_false / max_steps))
        self.writer.add_summary(summary, last_step)


    seen_step = -1
    paused = 0
    while paused < 360:
      print('start evaluation, model defined')
      if checkpoint_path:
        step = extract_step(checkpoint_path)
        last_checkpoint_path = checkpoint_path
      else:
        step, last_checkpoint_path = find_checkpoint(self.load_folder, seen_step)
      if step == -1:
        time.sleep(60)
        paused += 1
      else:
        paused = 0
        seen_step = step
        # Run Evaluation!
        run_evaluation(self, last_checkpoint_path)
        if checkpoint:
          break
