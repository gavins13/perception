#!python3
if __name__ == "__main__":
    print("running now...")
    ''' Config here '''
    experiment_id = 'tf2_test'
    experiment_type='train'
    ''' End Config  '''
    import matplotlib
    matplotlib.use("agg") #qt5agg

import os
import sys
import importlib
from lib_new.execution import Execution
from lib_new.dataset import Dataset
from experiments import experiments
from lib_new.misc import detect_cmd_arg, printt


def Experiment(experiment_id, experiment_type, execute=False, gpu=None, tensorboard_only=False):
    if gpu is not None:
        if not(isinstance(gpu, int)):
            raise ValueError('GPU number invalid')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        printt("GPU {} being used".format(gpu), warning=True)
    else:
        printt("GPU not being used", warning=True)
    
    if tensorboard_only is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        printt("Only Tensorboard starting", info=True)

    if not(experiment_type in ['train', 'test', 'evaluate']):
        raise ValueError('Execution mode invalid')

    if not(experiment_id in experiments.keys()):
        raise ValueError('Experiment doesn\'t exist')

    _tmp_mod_name = experiments[experiment_id]["module"]
    if experiment_type=='evaluate':
        if('eval_module' in experiments[experiment_id].keys()):
            _tmp_mod_name = experiments[experiment_id]["eval_module"]
    ArchFile = importlib.import_module(_tmp_mod_name)
    Model = ArchFile.Model

    if 'dataset_path' in experiments[experiment_id].keys():
        if 'dataset_path' in experiments[experiment_id].keys():
            sys.path.insert(0, experiments[experiment_id]['dataset_path'])
            print("Using user-specified dataset")
        _tmp_mod_name = experiments[experiment_id]["dataset"]
        dataset_module = importlib.import_module(_tmp_mod_name)
        Dataset = dataset_module.Dataset


    if 'experiment_name' in experiments[experiment_id].keys():
        experiment_name = experiments[experiment_id]["experiment_name"]
    else:
        experiment_name = 'TODELETE'



    if 'dataset_args' in experiments[experiment_id].keys():
        Dataset_Frame = Dataset(**experiments[experiment_id]['dataset_args'])
    else:
        Dataset_Frame = Dataset()

    Dataset = Dataset_Frame()
    # Dataset_Frame.create(); Dataset = Dataset_Frame

    if 'load_path' in experiments[experiment_id].keys():
        load_path = experiments[experiment_id]["load_path"]
    else:
        load_path = None

    if 'save_directory' in experiments[experiment_id].keys():
        save_directory = experiments[experiment_id]["save_directory"]
    else:
        save_directory = None


    return Execution(dataset=Dataset, experiment_name=experiment_name,
        load_path=load_path, model=Model, project_path=save_directory,
        experiment_type=experiment_type, execute=execute,
        tensorboard_only=tensorboard_only)





if __name__ == "__main__":
    experiment_id = detect_cmd_arg("experiment_id", false_val=experiment_id)
    experiment_type = detect_cmd_arg("type", false_val=experiment_type)
    gpu = detect_cmd_arg("gpu", false_val=None, val_dtype=int)
    tensorboard_only = detect_cmd_arg("tensorboard", retrieve_val=False)
    ThisExperiment = Experiment(experiment_id, experiment_type, execute=True, gpu=gpu, tensorboard_only=tensorboard_only)