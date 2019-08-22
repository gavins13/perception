from lib.multi_gpu_frame import multi_gpu_model as resources_model
from capsules.architectures.test import architecture as test_caps_architecture
from lib.data_frame import Data

from data.load_mnist import load_data
from lib.execution import execution

import tensorflow as tf

import os, sys
''' Config here '''
project_path = '/vol/biomedic/users/kgs13/PhD/capsule_networks/first_model'
cpu_only = True
num_gpus=1
'''
System is split into 3 objects:
 - An object which creates the capsule architecture and manages special operations.
 - An object which manages the system resources (e.g. GPUs) and divides the workload across available resources (GPUs), at the ending producing a suitable summary from all resources
 - An object which manages the training of the system, number of iterations run, and saving the model to the create points

NB/ in the future, the 'learning_core.py' might need to be rethought (as well as multi_gpu_frame.py) as it only really works for classification-type problems
'''

try:
  tf.enable_eager_execution()
  DataModel = Data(load_data, num_gpus=num_gpus)
  print("Start resource manager...")
  System = resources_model(cpu_only=cpu_only)
  print("Create Network Architecture...")
  CapsuleNetwork = test_caps_architecture()
  print("Strap Architecture to Resource Manager")
  System.strap_architecture(CapsuleNetwork)

  print("Strap Managed Architecture to a training scheme `Executer`")
  with execution(project_path, System, DataModel, experiment_name="test") as Executer:
      Executer.run_task(max_steps=1000, save_step=0)




except Exception as e:
  err_message = e.args
  print(err_message)
