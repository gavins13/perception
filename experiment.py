from lib.multi_gpu_frame import mutli_gpu_model as resources_model
from capsules.architectures import test as test_caps_architecture
from datetime import datetime
from lib.data import Data

from data.load_mnist import load
from lib.execution import execution

import os, sys
''' Config here '''
project_path = '/vol/biomedic/users/kgs13/PhD/capsule_networks/first_model'

'''
System is split into 3 objects:
 - An object which creates the capsule architecture and manages special operations.
 - An object which manages the system resources (e.g. GPUs) and divides the workload across available resources (GPUs), at the ending producing a suitable summary from all resources
 - An object which manages the training of the system, number of iterations run, and saving the model to the create points

NB/ in the future, the 'learning_core.py' might need to be rethought (as well as multi_gpu_frame.py) as it only really works for classification-type problems
'''

try:

  DataModel = Data(load, num_gpus=1)

  System = resources_model()
  CapsuleNetwork = test_caps_architecture()
  System.strap_architecture(CapsuleNetwork)

  with execution(project_path, System, DataModel) as Executer:
      Executer.run_task(max_steps=1000, save_step=0)




except Exception as e:
  err_message = e.args[0]
  print(err_message)
