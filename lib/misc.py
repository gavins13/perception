import collections
import tensorflow as tf
from pprint import pprint
from lib.execution import execution
from lib.multi_gpu_frame import multi_gpu_model as resources_model
from lib.data_frame import Data_V2 as Data
import sys

DataConfiguration = collections.namedtuple(
    "DataConfiguration",   ['project_path', 'execution_type',
                            'model_load_dir'])
SystemConfiguration = collections.namedtuple(
    "SystemConfiguration", ["cpu_only", "num_gpus",
                            "eager", "mini_batch_size", "validation_size",
                            "grow_memory"])


def run_model(load_data, Architecture, system_config, data_config, experiment_name, save_step=1, validation_step=5):
    try:
        if system_config.eager is True:
            tf.compat.v1.enable_eager_execution()
        DataModel = Data(load_data, num_gpus=system_config.num_gpus,
                         validation_size=system_config.validation_size)
        print("Start resource manager...")
        System = resources_model(cpu_only=system_config.cpu_only,
                                 eager=system_config.eager)
        print("Create Network Architecture...")
        Network = Architecture(evaluate=False if data_config.execution_type == 'train' else True)
        print("Strap Architecture to Resource Manager")
        System.strap_architecture(Network)

        print("Strap Managed Architecture to a training scheme `Executer`")
        with execution(data_config.project_path, System, DataModel,
                       experiment_name=experiment_name, max_steps_to_save=5,
                       mini_batch_size=system_config.mini_batch_size,
                       type=data_config.execution_type,
                       load=data_config.model_load_dir) as Executer:
            Executer.run_task(max_epochs=14000, save_step=save_step,
                              memory_growth=system_config.grow_memory, validation_step=validation_step)
    except Exception as e:
        err_message = e.args
        print("Exception thrown, see below:")
        print(err_message)
        pprint(e)


def detect_cmd_arg(arg, retrieve_val=True, val_dtype=str, false_val=None):
    '''
    This will look for the argument 'arg' in the Python command line input and if the retrieve_val is set to True, it will make sure to include an '=' sign in the input and retrieve the corresponding value

    If val_dtype is set, it will also perform a conversion on the argument to the specified data type
    '''
    try:
        assert isinstance(arg, str)
    except:
        raise ValueError("argument should be a string")
    arg = arg + "=" if retrieve_val is True else arg
    for i in range(len(sys.argv)):
        this_arg = sys.argv[i]
        if arg in this_arg:
            print(">>> Use a command-line given " + arg)
            if retrieve_val is True:
                val = this_arg.split(arg)
                val = val_dtype(val)
                print(arg + " CMD ARG DETECTED")
                print(arg)
                return val
            else:
                return True
    if retrieve_val is True:
        return false_val
    else:
        return False