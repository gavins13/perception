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
import random
import numpy as np


def Experiment(experiment_id, experiment_type='test', execute=False, gpu=None,
  tensorboard_only=False, reset=False, experiments_file=None, debug_level=None,
  gradient_taping=False, debug=False, command_line_arguments_enabled=True,
  metrics_enabled=False, metrics_printing_enabled=False, save_only=False,
  auto_gpu=False, perception_save_path=None, deterministic=False, set_seed=False,
  seed=None, validation_on_cpu=False, json_injection=None, ignore_json_evaluation_finished=False):
    '''
    experiment_id: (str) experiment_id from the JSON files
    experiment_type: (str) 'train' or 'evaluate'. Default: 'test'
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
        perception_save_path = detect_cmd_arg("perception_save_path", false_val=perception_save_path)
        ncpus = detect_cmd_arg("ncpus", false_val=None, val_dtype=int)
        ncpus = detect_cmd_arg("n_cpus", false_val=ncpus, val_dtype=int)
        deterministic = detect_cmd_arg("deterministic", retrieve_val=False, false_val=deterministic)
        set_seed = detect_cmd_arg("set_seed", retrieve_val=False, false_val=set_seed)
        seed_cmd = detect_cmd_arg("seed", false_val=None, val_dtype=int)
        validation_on_cpu = detect_cmd_arg("validation_on_cpu", retrieve_val=False, false_val=validation_on_cpu)
        ignore_json_evaluation_finished = detect_cmd_arg("ignore_json_evaluation_finished", retrieve_val=False, false_val=ignore_json_evaluation_finished)
        if seed_cmd is not None:
            set_seed = True
            seed = seed_cmd
    if deterministic is True:
        set_seed = True # []
    if (seed is None) and (set_seed is True):
        seed = 1114
    if deterministic is True:
        printt("Using deterministic kernels only", info=True)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        os.environ['TF_USE_CUDNN_AUTOTUNE'] = ''
    if set_seed is True:
        printt("Seeds set", info=True)
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)


    if ncpus is not None:
        printt("Limiting Number of CPUs to {}".format(ncpus))
        tf.config.threading.set_inter_op_parallelism_threads(ncpus)
        tf.config.threading.set_intra_op_parallelism_threads(ncpus)



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

    def injection(current, injectee):
        if current is None:
            return injectee
        if not(isinstance(current, dict)):
            return injectee
        else:
            if isinstance(injectee, dict):
                current = current.copy()
                for key in injectee.keys():
                    current[key] = injection(current.get(key), injectee[key])
                return current
            else:
                return injectee # or raise Error?
    if json_injection is not None:
        if not(isinstance(json_injection, dict)):
            printt("JSON Injection is not in a valid format", error=True, stop=True)
        for injection_key in json_injection.keys():
            experiments[experiment_id][injection_key] = injection(experiments[experiment_id].get(injection_key), json_injection[injection_key])
        printt("The following JSON data was injected into the existing experiment:", info=True)
        printt(json_injection, info=True)

    '''
    Set determinism/random seed properties
    '''
    set_seed_ = None
    if 'deterministic' in experiments[experiment_id].keys():
        if experiments[experiment_id]['deterministic'] is True:
            printt("Using deterministic kernels only", info=True)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
            os.environ['TF_USE_CUDNN_AUTOTUNE'] = ''
            set_seed_ = True
    if 'set_seed' in experiments[experiment_id].keys():
        set_seed_ = experiments[experiment_id]['set_seed']
    if 'seed' in experiments[experiment_id].keys():
        seed_ = experiments[experiment_id]['seed']
    elif set_seed_ is True:
        if seed is None:
            seed_ = 1114
        else:
            seed_ = seed
    if set_seed_ is True:
        printt("Seeds set", info=True)
        os.environ['PYTHONHASHSEED'] = str(seed_)
        random.seed(seed_)
        np.random.seed(seed_)
        tf.random.set_seed(seed_)

    '''
    Start loading modules
    '''
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

    if 'dataset' in experiments[experiment_id].keys():
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
    else:
        printt("Dataset not specified. Using Perception default.", warning=True)


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
    if perception_save_path is None:
        if 'perception_save_path' in experiments[experiment_id].keys():
            save_path = experiments[experiment_id]["perception_save_path"]
        else:
            save_path = None
    else:
        save_path = perception_save_path


    '''
    Experiment load weights folder name (relative to the perception save path
    or `save_path').

    This is useful when you want to load the weights of a pretrained model,
    and then resume training
    '''
    if 'load_weights_folder_name' in experiments[experiment_id].keys():
        load_weights_folder_name = experiments[experiment_id]["load_weights_folder_name"]
        printt("Load weights folder name: {}".format(load_weights_folder_name), info=True)
    else:
        load_weights_folder_name = None
        printt("No load weights folder from another experiment.", info=True)

    '''
    Reset optimisers?

    This is useful when you want to load the weights of a pretrained model,
    and then resume training but without the optimiser momentums.
    '''
    if 'reset_optimisers' in experiments[experiment_id].keys():
        reset_optimisers = experiments[experiment_id]["reset_optimisers"]
        printt("Optimisers to be reset according to JSON config", info=True)
    else:
        reset_optimisers = False
        printt("Optimiser state to be reloaded", info=True)


    return Execution(dataset=Dataset, experiment_name=experiment_name,
        save_folder=save_folder_name, model=Model,
        model_args=module_args,
        experiment_type=experiment_type, execute=execute,
        tensorboard_only=tensorboard_only, experiment_id=experiment_id,
        reset=reset, perception_save_path=save_path,
        experiments_manager=experiments, gradient_taping=gradient_taping,
        debug=debug, metrics_enabled=metrics_enabled,
        metrics_printing_enabled=metrics_printing_enabled, save_only=save_only,
        load_weights_folder_name=load_weights_folder_name,
        reset_optimisers=reset_optimisers, validation_on_cpu=validation_on_cpu,
        ignore_json_evaluation_finished=ignore_json_evaluation_finished)
