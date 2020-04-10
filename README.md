# Perception: A deep learning framework using TensorFlow

* Purely for learning purposes ;)

# Installation

## Dependencies (with Anaconda)
Install the following dependencies with Anaconda:
```
conda install python=3.7 tensorflow-gpu=2.0 matplotlib imageio=2.4.1=py37_0 ipython requests scipy pillow jupyter pandas
conda install -c conda-forge nibabel scikit-image moviepy tensorflow-probability
```

then check your pip location and version is correct (it should correspond to the Anaconda location)
```
pip --version
which pip
pip install --no-deps tensorflow-addons==0.6
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
