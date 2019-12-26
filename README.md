# Perception: A deep learning framework using TensorFlow

* Purely for learning purposes ;)

# Dependencies

```
conda install python=3.7 tensorflow-gpu matplotlib imageio=2.6.1=py37_0 ipython
conda install pillow
conda install jupyter
conda install -c conda-forge nibabel scikit-image

```
then check your pip location is correct (the conda location)
```
pip --version
which pip
pip install tensorflow-addons
```
# JSON "experiments.json" format

Each experiment entry has:
 - experiment_id (dictionary key)
 - experiment_name
 - dataset (optional)
 - dataset_path (optional)
 - dataset_args (optional)
 - save_folder (optional)
 - save_directory (optional)


# Issues
See ISSUES.md


# Adding experiments to experiments.py

Example of format for experiments.py:

```
    "tf2_test":
        {
            "module": 'models.test_model.main',
            "experiment_name": 'tf2_test',
            "dataset": "BiobankDataLoader",
            "dataset_path": "data/biobank", # if the data loader is in a different directory, an absolute path can be used
            "dataset_args":
                {
                    "cv_folds": 3, "cv_fold_number": 1 # Arguments to dictate the data splitting
                },
            "save_folder": None, # Folder will be automatically created for you
        },
```