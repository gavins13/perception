# Perception: A deep learning framework using TensorFlow

* Purely for learning purposes ;)

# Dependencies

```
conda install python=3.7 tensorflow-gpu matplotlib imageio=2.6.1=py37_0 ipython
conda install pillow
conda instal jupyter nibabel
```

## Current Issues:
1. Single GPU input pipeline is sorted but Multi-GPUs still has some performance issues with feeding in the data

## How to build architectures
Architecture models are objects derived from the base model in `lib/architecture.py`. An example of a architecture derived from this abstract class is shown in `models/example.py`.

## How to run a simple model
1. Import perception modules
```
import sys
from perception.lib.misc import DataConfiguration, SystemConfiguration, run_model
from perception.lib.architecture import Architecture as ABC_Architecture
```
2. Import your architecture/deep learning model
```
import MY_MODEL as Architecture
assert issubclass(Architecture, ABC_Architecture)
```
3. Import and run your data loader
```
import DATA_LOADER as Data_Loader
load_data = Data_Loader(**args)
```
4. Set your experiment name
```
experiment_name='MY EXPERIMENT NAME'
```
5. Set Data Configuration:
```
data_config = DataConfiguration(
    project_path='/vol/biomedic/users/kgs13/PhD/projects/misc_experiments',
    execution_type='train',
    model_load_dir=None)
```
6. Set System Configuration:
```
system_config = SystemConfiguration(cpu_only=False, num_gpus=4,
                                    eager=False, mini_batch_size=2,
                                    validation_size=1)
```
7. Run the model
```
run_model(Data, Architecture, system_config, data_config, experiment_name)
```

Appendix/Other
======
### Example code (brief)
Some example code of how to run this model is shown below
```
import sys
from perception.lib.misc import DataConfiguration, SystemConfiguration, run_model

''' Config here '''
sys.path.insert(
    0, '/homes/kgs13/biomedic/PhD/projects/project_name/architectures/')
from main_model import architecture as Architecture

from models.dc_cnn.architectures.main_aliased import architecture as Architecture

sys.path.insert(
    0, '/homes/kgs13/biomedic/PhD/projects/datasets/jose/data_loaders/')
from cine import load_data as Data_Loader


experiment_name = 'experiment_name_here'


data_config = DataConfiguration(
    project_path='/vol/biomedic/users/kgs13/PhD/projects/misc_experiments',
    execution_type='train',
    model_load_dir=None)


system_config = SystemConfiguration(cpu_only=False, num_gpus=4,
                                    eager=False, mini_batch_size=2,
                                    validation_size=1)
load_data = Data_Loader(dataset="acc=0.20")
''' End Config '''


run_model(Data, Architecture, system_config, data_config, experiment_name)
```
### Example code (comprehensive)
A more comprehensive version is shown below:
```
import sys
import tensorflow as tf
from pprint import pprint
from misc import DataConfiguration, SystemConfiguration
from lib.execution import execution
from lib.multi_gpu_frame import multi_gpu_model as resources_model
from lib.data_frame import Data_V2 as Data

''' Config here '''
from models.dc_cnn.architectures.main_aliased import architecture as Architecture
sys.path.insert(
    0, '/homes/kgs13/biomedic/PhD/projects/datasets/jose/data_loaders/')
from cine import load_data as Data_Loader
experiment_name = 'experiment_name_here'
data_config = DataConfiguration(
    project_path='/vol/biomedic/users/kgs13/PhD/projects/misc_experiments',
    execution_type='train',
    model_load_dir=None)
system_config = SystemConfiguration(cpu_only=False, num_gpus=4,
                                    eager=False, mini_batch_size=2,
                                    validation_size=1)
load_data = Data_Loader(dataset="acc=0.20")
''' End Config '''


try:
    if system_config.eager is True:
        tf.enable_eager_execution()
    DataModel = Data(load_data, num_gpus=system_config.num_gpus,
                     validation_size=system_config.validation_size)
    print("Start resource manager...")
    System = resources_model(cpu_only=system_config.cpu_only,
                             eager=system_config.eager)
    print("Create Network Architecture...")
    CapsuleNetwork = Architecture()
    print("Strap Architecture to Resource Manager")
    System.strap_architecture(CapsuleNetwork)

    print("Strap Managed Architecture to a training scheme `Executer`")
    with execution(data_config.project_path, System, DataModel,
                   experiment_name=experiment_name, max_steps_to_save=5,
                   mini_batch_size=system_config.mini_batch_size,
                   type=data_config.execution_type,
                   load=data_config.model_load_dir) as Executer:
        Executer.run_task(max_epochs=10000, save_step=1)
except Exception as e:
    err_message = e.args
    print("Exception thrown, see below:")
    print(err_message)
    pprint(e)

```

# Compiling TensorFlow with new ops in a Anaconda virtual environment
1. Clone and checkout require TF release and make CUDA ops changes. Remember to update the BUILD files to include required kernels.
2. Create new conda virtual environment. Run "conda install python3.6" (To speed up process can create and install tensorflow-gpu)
3. In repo, run "./configure"
4. The python version is found run running "which python3.6" inside the environment. You should use python3.6 (install it if not present)
5. The python library is the one in the anaconda environment directory
6. The CUDA version is 9, and directory is /vol/cuda/9.0.176
7. CUDNN version is 7 and directory is also /vol/cuda/9.0.176
8. Answer no/enter to everything else
9. Run "PYTHONPATH= bazel --output_base=/vol/biomedic/users/kgs13/Software/bazel_base build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package"
10. "mkdir ../tmp"
11. "PYTHONPATH= ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ../tmp/tensorflow_pkg"
12. cd ../tmp/tensorflow_pkg
13. PYTHONPATH= pip install tensorflow-1.12.2-cp36-cp36m-linux_x86_64.whl
14. Can test by doing "PYTHONPATH= python3.6" and then "import tensorflow"



# Running with conda

```
conda activate tf_roll
```
```
CUDA_VISIBLE_DEVICES=3 PYTHONPATH= CUDA_PATH=/vol/biomedic/users/kgs13/Software/cuda_envs/9.2.148-cudnn7.2.1/cuda CUDA_INC_PATH=/vol/biomedic/users/kgs13/Software/cuda_envs/9.2.148-cudnn7.2.1/cuda/include LD_LIBRARY_PATH=/vol/biomedic/users/kgs13/Software/cuda_envs/9.2.148-cudnn7.2.1/cuda/lib64:/vol/biomedic/users/kgs13/Software/cuda_envs/9.2.148-cudnn7.2.1/cuda/:$LD_LIBRARY_PATH python3 experiment.py
```
