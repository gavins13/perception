from lib.multi_gpu_frame import multi_gpu_model as resources_model
from lib.data_frame import Data
import os, sys
import tensorflow as tf
import collections
#from data.load_mnist import load_data_light as load_data
sys.path.insert(0, '/homes/kgs13/biomedic/PhD/sunnybrook-data-dicom/')
from load_data import load_data
from lib.execution import execution
DataConfiguration = collections.namedtuple("DataConfiguration", ['project_path', 'execution_type'])
SystemConfiguration = collections.namedtuple("SystemConfiguration", ["cpu_only", "num_gpus", "eager", "mini_batch_size"])

''' Config here '''
from capsules.architectures.v0_rev0_1 import architecture as architecture
from capsules.architectures.v0_rev1_2_test import architecture as test_architecture
data_config = DataConfiguration('/vol/biomedic/users/kgs13/PhD/capsule_networks/first_model','train')
system_config = SystemConfiguration(False, num_gpus=1, False, mini_batch_siz=4)
''' End Config '''

try:
  if data_config.eager==True:
    tf.enable_eager_execution()
  DataModel = Data(load_data, num_gpus=system_config.num_gpus)
  print("Start resource manager...")
  System = resources_model(cpu_only=system.config.cpu_only,eager=system.config.eager)
  print("Create Network Architecture...")
  if(test==True):
      print("Running Test Architecture... *** MAIN ARCHITECTURE NOT BEING USED ***")
      CapsuleNetwork = test_architecture()
  else:
      CapsuleNetwork = architecture()
  print("Strap Architecture to Resource Manager")
  System.strap_architecture(CapsuleNetwork)

  print("Strap Managed Architecture to a training scheme `Executer`")
  with execution(project_path, System, DataModel, experiment_name="test", max_steps_to_save=1000, mini_batch_size=mini_batch_size, type=execution_type) as Executer:
      Executer.run_task(max_steps=1000, save_step=1)
except Exception as e:
  err_message = e.args
  print(err_message)
