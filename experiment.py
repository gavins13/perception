from lib.multi_gpu_frame import multi_gpu_model as resources_model
from capsules.architectures.test import architecture as test_caps_architecture
from lib.data_frame import Data

from data.load_mnist import load_data_light as load_data
from lib.execution import execution

import tensorflow as tf

import os, sys
''' Config here '''
project_path = '/vol/biomedic/users/kgs13/PhD/capsule_networks/first_model'
cpu_only = False
num_gpus=1
eager=False

''' Logging '''
import logging

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
fh = logging.FileHandler(project_path + '/logs/' + 'tensorflow.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

'''
System is split into 3 objects:
 - An object which creates the capsule architecture and manages special operations.
 - An object which manages the system resources (e.g. GPUs) and divides the workload across available resources (GPUs), at the ending producing a suitable summary from all resources
 - An object which manages the training of the system, number of iterations run, and saving the model to the create points

NB/ in the future, the 'learning_core.py' might need to be rethought (as well as multi_gpu_frame.py) as it only really works for classification-type problems
'''

try:
  if eager==True:
    tf.enable_eager_execution()
  DataModel = Data(load_data, num_gpus=num_gpus)
  print("Start resource manager...")
  System = resources_model(cpu_only=cpu_only,eager=eager)
  print("Create Network Architecture...")
  CapsuleNetwork = test_caps_architecture()
  print("Strap Architecture to Resource Manager")
  System.strap_architecture(CapsuleNetwork)

  print("Strap Managed Architecture to a training scheme `Executer`")
  #with execution(project_path, System, DataModel, experiment_name="test") as Executer:
  #    Executer.run_task(max_steps=1000, save_step=0)
  Executer = execution(project_path, System, DataModel, experiment_name="test", max_steps_to_save=1000)
  Executer.__enter__()
  Executer.run_task(max_steps=1000, save_step=1)
  Executer.__exit__(None,None,None)



except Exception as e:
  err_message = e.args
  print(err_message)
