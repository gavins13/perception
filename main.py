print("running now...")
import sys
import matplotlib
matplotlib.use("agg") #qt5agg
import importlib
from lib_new.execution import Execution
from lib_new.dataset import Dataset
from experiments import experiments


''' Config here '''
experiment_id = 'MERUPv4_magnitude_v2'
experiment_type='train'
''' End Config  '''



experiment_id = detect_cmd_arg("experiment_id", false_val=experiment_id)
experiment_type = detect_cmd_arg("type", false_val=experiment_type)


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


if __name__ == "main":
	Experiment = Execution(dataset=Dataset, experiment_name=experiment_name,
		load_path=load_path, model=Model, project_path=save_directory,
		experiment_type=experiment_type, execute=True)
else:
	Experiment = Execution(dataset=Dataset, experiment_name=experiment_name,
		load_path=load_path, model=Model, project_path=save_directory,
		experiment_type=experiment_type)
	# e.g. from main import Experiment