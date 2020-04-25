#!python
import os
from .lib.misc import printt
from .lib.execution import Execution
from .lib.dataset import Dataset
from .lib.experiments import Experiments as ExperimentsManager
from .lib.misc import detect_cmd_arg, path as prepare_path
import sys
import importlib
import tensorflow as tf


def Experiment(experiment_id, experiment_type='test', execute=False, gpu=None,
  tensorboard_only=False, reset=False, experiments_file=None, debug_level=None,
  gradient_taping=False, debug=False, command_line_arguments_enabled=True,
  metrics_enabled=False, metrics_printing_enabled=False, save_only=False,
  auto_gpu=False):
    '''
    experiment_id: (str) experiment_id from the JSON files
    experiment_type: (str) 'train' or 'test'. Default: 'test'
    (optional) execute: (bool) Execute before returning?. Default: False
    (optional) gpu: (int) Which GPU number to use
    (optional) tensorboard_only: (bool) Only start TensorBoard (TB)? Default: False (i.e. start TB if training, otherwise do not start)
    (optional) reset: (bool) train from scratch? Default: False
    (optional) gradient_taping: (bool) Using GradientTape or train with Keras API? Use GradientTape (True) for easier debugging. Default: False
    (optional) debug: (bool) If True, then loss_func doesn't use the tf.function wrapper (for graph building) and if gradient_taping is False (i.e. using Keras API) then it forces the Keras model compiler to run_eagerly. Default: False.
    (optional) command_line_arguments_enabled: (bool) Allow options to specified from command arguments. Default: True

    Other command line arguments:
    --debug-level: (int) From 0 to 5. 0: Print nothing. 1: Print errors. 2: Print warnings. 3: Print Information. 4: Print Debug Information 5: Print Everything

    NOTE: the command line argument 'debug_level' is not affected by the command_line_arguments_enabled option

    NOTE: for full debugging mode, gradient_taping should be True, and debug should be True. Future versions of Perception will probably automatically switch gradient_taping to True when debug is True.

    NOTE: if using with a High Performance Computing cluster or Collab, then please use auto_gpu=True to leave it to the parent execution node to decide which GPU to select (e.g. via PBS)
    '''

    if command_line_arguments_enabled is True:
        experiment_id = detect_cmd_arg("experiment_id", false_val=experiment_id)
        experiment_id = detect_cmd_arg("experiment", false_val=experiment_id)
        experiments_file = detect_cmd_arg("experiments_file", false_val=experiments_file)
        experiment_type = detect_cmd_arg("type", retrieve_val=True, false_val=experiment_type, val_dtype=str)
        gpu = detect_cmd_arg("gpu", false_val=gpu, val_dtype=int)
        tensorboard_only = detect_cmd_arg("tensorboard", retrieve_val=False, false_val=tensorboard_only)
        tensorboard_only_2 = detect_cmd_arg("tensorboard_only", retrieve_val=False, false_val=tensorboard_only)
        tensorboard_only = (tensorboard_only or tensorboard_only_2)
        reset = detect_cmd_arg("reset", retrieve_val=False, false_val=reset)
        gradient_taping = detect_cmd_arg("gradient_taping", retrieve_val=False, false_val=gradient_taping)
        debug = detect_cmd_arg("debug", retrieve_val=False, false_val=debug)
        metrics_enabled = detect_cmd_arg("metrics_enabled", retrieve_val=False, false_val=metrics_enabled)
        metrics_enabled = detect_cmd_arg("metrics", retrieve_val=False, false_val=metrics_enabled)
        metrics_printing_enabled = detect_cmd_arg("metrics_printing_enabled", retrieve_val=False, false_val=metrics_printing_enabled)
        metrics_printing_enabled = detect_cmd_arg("metrics_printing", retrieve_val=False, false_val=metrics_printing_enabled)
        save_only = detect_cmd_arg("save", retrieve_val=False, false_val=save_only)
        save_only_2 = detect_cmd_arg("save_only", retrieve_val=False, false_val=save_only)
        save_only = (save_only or save_only_2)
        auto_gpu = detect_cmd_arg("auto_gpu", retrieve_val=False, false_val=auto_gpu)



    experiments = ExperimentsManager(experiments_file=experiments_file)
    if tensorboard_only is True:
        gpu = None
        reset = False
        printt("Only Tensorboard starting", info=True)

    if save_only is True:
        reset = False
        printt("Only saving the model", info=True)

    if debug_level is not None:
        if not(isinstance(debug_level, int)):
            raise ValueError('Invalid DEBUG_LEVEL')
        os.environ["PYTHON_PERCEPTION_DEBUG_LEVEL"] = str(debug_level)
        printt("DEBUG LEVEL {} being used".format(debug_level), info=True)

    if auto_gpu is False:
        if gpu is not None:
            if not(isinstance(gpu, int)):
                raise ValueError('GPU number invalid')
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            printt("GPU {} being used".format(gpu), warning=True)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ''
            printt("GPU not being used", warning=True)

    memory_growth = detect_cmd_arg("memory_growth", retrieve_val=False, false_val=False)
    if memory_growth is True:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


    if not(experiment_type in ['train', 'test', 'testing', 'evaluate']):
        printt('Execution mode invalid', error=True, stop=True)

    if not(experiment_id in experiments.keys()):
        printt("Experiment doesn't exist", error=True, stop=True)

    if 'module' in experiments[experiment_id].keys():
        if 'module_path' in experiments[experiment_id].keys():
            sys.path.insert(0, prepare_path(
                experiments[experiment_id]['module_path'])
            )
            printt("Using user-specified module path", info=True)
        _tmp_mod_name = experiments[experiment_id]["module"]
        if experiment_type in ['evaluate', 'test']:
            if('eval_module' in experiments[experiment_id].keys()):
                _tmp_mod_name = experiments[experiment_id]["eval_module"]
        ArchFile = importlib.import_module(_tmp_mod_name)
        Model = ArchFile.Model
    else:
        printt("Architecture/Model not specified", error=True, stop=True)


    if 'module_args' in experiments[experiment_id].keys():
        module_args = experiments[experiment_id]['module_args']
    else:
        module_args = None

    if 'dataset_path' in experiments[experiment_id].keys():
        if 'dataset_path' in experiments[experiment_id].keys():
            printt("DATASET PATH: "+prepare_path(
                experiments[experiment_id]['dataset_path']), debug=True)
            sys.path.insert(0, prepare_path(
                experiments[experiment_id]['dataset_path'])
            )
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

    '''
    Experiment save path (relative to the perception save path or `save_path')
    '''
    if 'save_folder_name' in experiments[experiment_id].keys():
        save_folder_name = experiments[experiment_id]["save_folder_name"]
        printt("Save folder path: {}".format(save_folder_name), info=True)
    else:
        save_folder_name = None
        printt("No save folder. Creating one for you...", info=True)

    '''
    Perception save path
    '''
    if 'save_path' in experiments[experiment_id].keys():
        save_path = experiments[experiment_id]["save_path"]
    else:
        save_path = None


    return Execution(dataset=Dataset, experiment_name=experiment_name,
        save_folder=save_folder_name, model=Model,
        model_args=module_args,
        experiment_type=experiment_type, execute=execute,
        tensorboard_only=tensorboard_only, experiment_id=experiment_id,
        reset=reset, perception_save_path=save_path,
        experiments_manager=experiments, gradient_taping=gradient_taping,
        debug=debug, metrics_enabled=metrics_enabled,
        metrics_printing_enabled=metrics_printing_enabled, save_only=save_only)
