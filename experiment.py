from lib.multi_gpu_frame import multi_gpu_model as resources_model
from capsules.architectures.v0_rev0_1 import architecture as architecture
from capsules.architectures.v0_rev1_2_test import architecture as test_architecture
from lib.data_frame import Data

import os, sys
import tensorflow as tf


#from data.load_mnist import load_data_light as load_data
sys.path.insert(0, '/homes/kgs13/biomedic/PhD/sunnybrook-data-dicom/')
from load_data import load_data

from lib.execution import execution

''' Config here '''
project_path = '/vol/biomedic/users/kgs13/PhD/capsule_networks/first_model'
cpu_only = False
num_gpus=1
eager=False
test=True
mini_batch_size=4

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
  if(test==True):
      print("Runnin Test Architecture... *** MAIN ARCHITECTURE NOT BEING USED ***")
      CapsuleNetwork = test_architecture()
  else:
      CapsuleNetwork = architecture()
  print("Strap Architecture to Resource Manager")
  System.strap_architecture(CapsuleNetwork)

  print("Strap Managed Architecture to a training scheme `Executer`")
  #with execution(project_path, System, DataModel, experiment_name="test") as Executer:
  #    Executer.run_task(max_steps=1000, save_step=0)
  Executer = execution(project_path, System, DataModel, experiment_name="test", max_steps_to_save=1000, mini_batch_size=mini_batch_size)
  Executer.__enter__()
  Executer.run_task(max_steps=1000, save_step=1)
  Executer.__exit__(None,None,None)



except Exception as e:
  err_message = e.args
  print(err_message)
