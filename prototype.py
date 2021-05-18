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


def Experiment(Model, Dataset, experiment_name=None, gpu=None,
  metrics_enabled=False, metrics_printing_enabled=False,
  auto_gpu=True, perception_save_path=None, dir=None, deterministic=False, set_seed=False, debug=False, debug_level=None,
  seed=None, validation_on_cpu=False,
  tensorboard_port=None, ncpus=None, memory_growth=False,
  save_folder_name=None, load_weights_folder_name=None, reset_optimisers=False, gradient_taping=False, **kwargs):
    '''
    '''

    if experiment_name is None:
        try:
            experiment_name_ = Model.__name__
        except:
            experiment_name_ = type(Model).__name__
    else:
        experiment_name_ = experiment_name
  
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


    if debug_level is not None:
        if not(isinstance(debug_level, int)):
            raise ValueError('Invalid DEBUG_LEVEL')
        os.environ["PYTHON_PERCEPTION_DEBUG_LEVEL"] = str(debug_level)
        printt("DEBUG LEVEL {} being used".format(debug_level), info=True)

    
    if gpu is not None:
        if not(isinstance(gpu, int)):
            raise ValueError('GPU number invalid')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        printt("GPU {} being used".format(gpu), warning=True)
    else:
        if auto_gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ''
            printt("GPU not being used", warning=True)
        else:
            printt(
                "Inherit CUDA_VISIBLE_DEVICES. Value: {}".format(
                    os.environ['CUDA_VISIBLE_DEVICES']
                )
            )

    if memory_growth is True:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    '''
    Perception save path
    '''
    if perception_save_path is None:
        if dir is None:
            save_path = None
        else:
            save_path = dir
    else:
        save_path = perception_save_path


    return Execution(dataset=Dataset, experiment_name=experiment_name_,
        save_folder=save_folder_name, model=Model,
        experiment_type='train', execute=False,
        tensorboard_only=False, experiment_id=None,
        reset=None, perception_save_path=save_path,
        experiments_manager=None, gradient_taping=gradient_taping,
        debug=debug, metrics_enabled=metrics_enabled,
        metrics_printing_enabled=metrics_printing_enabled, save_only=False,
        load_weights_folder_name=load_weights_folder_name,
        reset_optimisers=reset_optimisers, validation_on_cpu=validation_on_cpu,
        ignore_json_evaluation_finished=None,
        tensorboard_port=tensorboard_port, model_args=kwargs)
