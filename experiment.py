from lib.multi_gpu_frame import multi_gpu_model as resources_model
from lib.data_frame import Data
import os, sys
import tensorflow as tf
import collections
from pprint import pprint
#from data.load_mnist import load_data_light as load_data
sys.path.insert(0, '/homes/kgs13/biomedic/PhD/sunnybrook-data-dicom/')
from load_data import load_data as Data_Loader
from load_data_augmented import load_data as Augmented_Data_Loader
from lib.execution import execution
DataConfiguration = collections.namedtuple("DataConfiguration", ['project_path', 'execution_type', 'model_load_dir'])
SystemConfiguration = collections.namedtuple("SystemConfiguration", ["cpu_only", "num_gpus", "eager", "mini_batch_size", "test_architecture", "validation_size"])

''' Config here '''
from capsules.architectures.v0_rev0_1 import architecture as Architecture
from capsules.architectures.v1_rev4_1 import architecture as TestArchitecture
experiment_name = 'v1_rev4_1'
data_config = DataConfiguration(project_path='/vol/biomedic/users/kgs13/PhD/capsule_networks/first_model',execution_type='train',
    model_load_dir='v1_rev4_1_2018-07-13-17:22:41.612039')
system_config = SystemConfiguration(cpu_only=False, num_gpus=1, eager=False, mini_batch_size=2, test_architecture=True, validation_size=2)
load_data = Data_Loader(dataset="acc=0.29")
#load_data = Augmented_Data_Loader(dataset="acc=0.29", load_from_saved=True, save_path='/vol/biomedic/users/kgs13/PhD/sunnybrook-data-dicom/saved_datasets.p/augmented', save=True)
''' End Config '''
# v1_rev1_3_2018-07-04-16:00:27.537862(basereconcaps)  v1_rev1_2_2018-07-04-12:11:47.413890(deepreconcaps) baseline_v1_rev1_3_2018-07-10-12:39:28.799803(baseline for reconcaps) v1_rev3_1_2018-07-04-21:32:34.241743(iterativecaps) v1_rev4_1_2018-07-13-17:22:41.612039(3x3 convcps) v1_rev4_2_2018-07-17-15:23:33.786890(5x5 convcaps mb=2)     baseline_v1_rev1_3_nocaps_2018-07-17-17:46:53.833824(basereconcaps without caps - attemp2: v1_rev4_2_2018-07-17-20:16:35.964292)   SR_v1_rev4_1_2018-07-18-14:04:32.664220(3x3 concaps) 'v1_rev4_2_2018-07-19-10:47:41.929848' 'v1_rev4_2_2018-07-19-21:02:49.300450'   v1_rev4_3_2018-07-20-16:02:49.919580(new 5x5 architecture, deeper conv for capsule input)
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
        Executer.run_task(max_epochs=10000, save_step=1)
except Exception as e:
  err_message = e.args
  print("Exception thrown, see below:")
  print(err_message)
  pprint(e)
