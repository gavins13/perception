from lib.multi_gpu_frame import multi_gpu_model as resources_model
from lib.data_frame import Data
import os, sys
import tensorflow as tf
import collections
from pprint import pprint
#from data.load_mnist import load_data_light as load_data
sys.path.insert(0, '/homes/kgs13/biomedic/PhD/sunnybrook-data-dicom/')
from load_data import load_data
from lib.execution import execution
DataConfiguration = collections.namedtuple("DataConfiguration", ['project_path', 'execution_type', 'model_load_dir'])
SystemConfiguration = collections.namedtuple("SystemConfiguration", ["cpu_only", "num_gpus", "eager", "mini_batch_size", "test_architecture", "validation_size"])

''' Config here '''
from capsules.architectures.v0_rev0_1 import architecture as Architecture
from capsules.architectures.v0_rev13_1_verylowparams_9_residual_highres import architecture as TestArchitecture
experiment_name = 'v0_rev13_1_verylowparams_9_residual_highres'
data_config = DataConfiguration(project_path='/vol/biomedic/users/kgs13/PhD/capsule_networks/first_model',execution_type='train',
    model_load_dir=None)
system_config = SystemConfiguration(cpu_only=False, num_gpus=1, eager=False, mini_batch_size=4, test_architecture=True, validation_size=1)
''' End Config '''

try:
  if system_config.eager==True:
    tf.enable_eager_execution()
  DataModel = Data(load_data, num_gpus=system_config.num_gpus, validation_size=system_config.validation_size)
  print("Start resource manager...")
  System = resources_model(cpu_only=system_config.cpu_only,eager=system_config.eager)
  print("Create Network Architecture...")
  if(system_config.test_architecture==True):
      print("Running Test Architecture... *** MAIN ARCHITECTURE NOT BEING USED ***")
      CapsuleNetwork = TestArchitecture()
  else:
      CapsuleNetwork = Architecture()
  print("Strap Architecture to Resource Manager")
  System.strap_architecture(CapsuleNetwork)

  print("Strap Managed Architecture to a training scheme `Executer`")
  with execution(data_config.project_path, System, DataModel, experiment_name=experiment_name, max_steps_to_save=5, mini_batch_size=system_config.mini_batch_size, type=data_config.execution_type, load=data_config.model_load_dir) as Executer:
        Executer.run_task(max_steps=1000, save_step=1)
except Exception as e:
  err_message = e.args
  print("Exception thrown, see below:")
  print(err_message)
  pprint(e)
