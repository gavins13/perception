#!python
import os
if __name__ == "__main__":
    from lib.misc import printt
    print("running now...")
    import matplotlib
    matplotlib.use("agg") #qt5agg
    import json
    if os.path.isfile("Config.perception") is True:
        with open("Config.perception", "r") as config_file:
            Config = json.load(config_file)
    else:
        printt("Config.perception doesn't exist", error=True, stop=True)
    experiment_id = Config['defaults']['experiment_id']
    experiment_type = Config['defaults']['experiment_type']
    from lib.execution import Execution
    from lib.dataset import Dataset
    from lib.experiments import Experiments as ExperimentsManager
    from lib.misc import detect_cmd_arg
else:
    from .lib.misc import printt
    from .lib.execution import Execution
    from .lib.dataset import Dataset
    from .lib.experiments import Experiments as ExperimentsManager
    from .lib.misc import detect_cmd_arg

import sys
import importlib


def Experiment(experiment_id, experiment_type='test', execute=False, gpu=None,
  tensorboard_only=False, reset=False):
    '''
    experiment_id: (str) experiment_id from the JSON files
    experiment_type: (str) 'train' or 'test'
    (optional) execute: (bool) Execute before returning?
    (optional) gpu: (int) Which GPU number to use
    (optional) tensorboard_only: (bool) Only start TensorBoard (TB)? Default is to start TB with training
    (optional) reset: (bool) train from scratch?
    '''
    experiments = ExperimentsManager()
    if tensorboard_only is True:
        gpu = None
        printt("Only Tensorboard starting", info=True)

    if gpu is not None:
        if not(isinstance(gpu, int)):
            raise ValueError('GPU number invalid')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        printt("GPU {} being used".format(gpu), warning=True)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        printt("GPU not being used", warning=True)

    if not(experiment_type in ['train', 'test', 'evaluate']):
        printt('Execution mode invalid', error=True, stop=True)

    if not(experiment_id in experiments.keys()):
        printt("Experiment doesn't exist", error=True, stop=True)

    if 'module' in experiments[experiment_id].keys():
        if 'module_path' in experiments[experiment_id].keys():
            sys.path.insert(0, experiments[experiment_id]['dataset_path'])
            printt("Using user-specified module path", info=True)
        _tmp_mod_name = experiments[experiment_id]["module"]
        if experiment_type in ['evaluate', 'test']:
            if('eval_module' in experiments[experiment_id].keys()):
                _tmp_mod_name = experiments[experiment_id]["eval_module"]
        ArchFile = importlib.import_module(_tmp_mod_name)
        Model = ArchFile.Model
    else:
        printt("Architecture/Model not specified", error=True, stop=True)

    if 'dataset_path' in experiments[experiment_id].keys():
        if 'dataset_path' in experiments[experiment_id].keys():
            sys.path.insert(0, experiments[experiment_id]['dataset_path'])
            print("Using user-specified dataset path")
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

    if 'save_folder_name' in experiments[experiment_id].keys():
        save_folder_name = experiments[experiment_id]["save_folder_name"]
    else:
        save_folder_name = None

    if 'save_path' in experiments[experiment_id].keys():
        save_path = experiments[experiment_id]["save_path"]
    else:
        save_path = None


    return Execution(dataset=Dataset, experiment_name=experiment_name,
        save_folder=save_folder_name, model=Model,
        experiment_type=experiment_type, execute=execute,
        tensorboard_only=tensorboard_only, experiment_id=experiment_id,
        reset=reset, perception_save_path=save_path)





if __name__ == "__main__":
    experiment_id = detect_cmd_arg("experiment_id", false_val=experiment_id)
    experiment_id = detect_cmd_arg("experiment", false_val=experiment_id)
    experiment_type = detect_cmd_arg("type", false_val=experiment_type)
    gpu = detect_cmd_arg("gpu", false_val=None, val_dtype=int)
    tensorboard_only = detect_cmd_arg("tensorboard", retrieve_val=False)
    tensorboard_only_2 = detect_cmd_arg("tensorboard_only", retrieve_val=False)
    tensorboard_only = (tensorboard_only or tensorboard_only_2)
    reset = detect_cmd_arg("reset", retrieve_val=False)
    ThisExperiment = Experiment(experiment_id, experiment_type, execute=True, gpu=gpu, tensorboard_only=tensorboard_only, reset=reset)
