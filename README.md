# Perception: A deep learning framework using TensorFlow

* Purely for learning purposes ;)

# Installation

## Dependencies (with Anaconda)
Install the following dependencies with Anaconda:
```
conda install python=3.7 tensorflow-gpu=2.0 matplotlib imageio=2.4.1=py37_0 ipython requests scipy pillow jupyter pandas tabulate
conda install -c conda-forge nibabel scikit-image moviepy tensorflow-probability
```

then check your pip location and version is correct (it should correspond to the Anaconda location)
```
pip --version
which pip
pip install --no-deps tensorflow-addons==0.6
```

### Other Dependencies that you might use:

NLTK
```
conda install -c anaconda nltk
```

SkLearn
```
conda install -c anaconda scikit-learn
```

## Installation Issues
If issues with tensorflow-addons arises, please ignore the instructions above and install this way:
```
... activate conda environment ...
conda install python=3.7 matplotlib imageio=2.4.1=py37_0 ipython requests scipy pillow jupyter pandas tabulate
conda install -c conda-forge nibabel scikit-image moviepy
export CUDA_HOME=/vol/cuda/10.1.105-cudnn7.6.5.32 # Insert your CUDA path
export PATH=${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export CPATH=${CUDA_HOME}/include:$CPATH
which pip # check conda pip is being used
pip install tensorflow tensorflow-addons tensorflow-probability tensorboard=2.1.0 # May need to select tensorboard 2.1.0
```
Test with:
```
import tensorflow as tf, numpy as np, tensorflow_addons as tfa
a = tf.zeros([1,256,256,3])
b = tfa.image.translate(a, [[5,4]])
```

You may receive the error 'libnvinfer.so.6' or 'libnvinfer_plugin.so.6' cannot be found regarding TensorRT. This is address here: https://github.com/tensorflow/tensorflow/issues/35968.
Otherwise, you may downgrade to tensorflow==2.0.0



### Patches (important)

Please run install.py via your conda python installation. Otherwise, you can apply the patches manually below:

Reproducibility and potentially unexpected behaviour can occur due to non-deterministic GPU calcuations on backprop of the `tf.gather()` function. Please see [Issue 39751](https://github.com/tensorflow/tensorflow/issues/39751). In order to fix this, please follow the following instructions.

1. Locate `array_grad.py` in `tensorflow/python/ops`
2. Find and replace `math_ops.unsorted_segment_sum` with `unsorted_segment_sum_fix`.
3. Copy and paste the following code after package imports:
```
def unsorted_segment_sum_fix(*args, **kwargs):
  args = list(args)
  params = args[0]
  orig_dtype = params.dtype
  if 'float' in str(params.dtype):
    params = math_ops.cast(params, dtype=dtypes.float64)
  elif 'complex' in str(params.dtype):
    params = math_ops.cost(params, dtype=dtypes.complex128)
  args[0] = params
  result = math_ops.unsorted_segment_sum(*args, **kwargs)
  result = math_ops.cast(result, dtype=orig_dtype)
  return result
```



## As a package (optional)
To install perception as a package, move the directory containing this file to your local python "site-packages" or alternatively

## JSON "experiments.json" format

Each experiment entry has:
 - experiment_id (dictionary key)
 - experiment_name
 - dataset (optional)
 - dataset_path (optional)
 - dataset_args (optional)
 - save_folder (optional)
 - save_directory (optional)

## Config.perception format (JSON format)
Specify the default experiment ID, save directory and other variables here.
Typical format shown below:
```
{
  "defaults": {
    experiment_id: 'tf2_test',
    experiment_type: 'train'
  },
	"save_directory": "/vol/biomedic/users/kgs13/PhD/projects/misc_experiments/tf2_experimental_results/"
}
```
# Perception main.py arguments.py
'''
from experiment import debug_level
debug_level:
0 = No printing except execution
1 = Warnings
2 = Errors
3 = Information
4 = Debugging Information
5 = All (Full debugging)
'''


# Issues
See ISSUES.md


# Adding experiments to experiments.py

Example of format for experiments.py:

```
    "tf2_test":
        {
            "module": "models.test_model.main",
            "experiment_name": "tf2_test",
            "dataset": "BiobankDataLoader", # Module name. Absolute and Relative imports are both supported
            "dataset_path": "data/biobank", # if the data loader is in a different directory, an absolute path can be used
            "dataset_args":
                {
                    "cv_folds": 3, "cv_fold_number": 1 # Arguments to dictate the data splitting
                },
            "save_folder_name": null, # Folder will be automatically created for you
            "save_path": "/absolute/path/to/location/containing/save_folder_name" # aka 'perception_save_path'
        },
```

In the specification of the "dataset_path" or "module_path", you can use the string "**__path__**" which will be used to fetch the perception install directory. This is used in the case where you wish to place modules in the perception directory but have not updated package imports or initialised packages
