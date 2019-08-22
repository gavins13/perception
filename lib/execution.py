import tensorflow as tf

import os, sys

from datetime import datetime

# Object is a context manager!
class execution(object):
  def __init__(self, project_path, model, data_strap, type='train', load=None, experiment_name=None):
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

    # Set up the data
    self.data_strap = data_strap
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

    # Set up the retrieval of the results (I.e. Build the Summarisd results, and results Graph)
    with tf.Graph().as_default():
    #   -->  Build model
        print(">> Time to build TF Graph!")
        self.summarised_result, self.results = model.run_multi_gpu(data_strap)
    #   -->  Print stats
    self.param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.
        TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    sys.stdout.write('total_params: %d\n' % self.param_stats.total_parameters)

  def __enter__(self):
      print(">Create TF FileWriter")
      self.writer = tf.summary.FileWriter(self.summary_folder)

  def run_task(self, max_steps, save_step=1, max_steps_to_save=1000):
      # save_step defines the increment amount before saving a new checkpoint
      print(">Create TF session")
      self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
      init_op = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())
      print(">Initialise sesssion with variables")
      self.session.run(init_op) # Initialise graph with variables
      print(">Create TF Saver")
      self.saver = tf.train.Saver(max_to_keep=max_steps_to_save)
      print(">Load last saved model")
      self.last_step = self.load_saved_model()
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=session, coord=coord)
      try:
        self.experiment(max_steps=max_steps,save_step=save_step)
      except tf.errors.OutOfRangeError:
        tf.logging.info('Finished experiment.')
      finally:
        coord.request_stop()
      coord.join(threads)
      session.close()


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
          tf.gfile.DeleteRecursively(load_dir)
          tf.gfile.MakeDirs(load_dir)
          prev_step = 0
    else:
        tf.gfile.MakeDirs(load_dir)
        prev_step = 0
    return prev_step

  def __exit__(self):
    self.writer.close()

  # Task - Training
  def training(self, max_steps, save_step):
    step = 0
    for i in range(self.last_step, max_steps):
        print("training: %d" % i)
        step += 1
        summary, _ = self.session.run([self.result.summary, self.result.train_op]) # Run graph
        self.writer.add_summary(summary, i)
        if (i + 1) % save_step == 0:
          self.saver.save(self.session, os.path.join(self.summary_folder, 'model.ckpt'), global_step=i + 1)


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
