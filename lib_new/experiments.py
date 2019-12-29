import json
from os import listdir
from os.path import isfile, join
from lib_new.misc import printt

exp_ext = '.json'
json_files = [join('experiments/', f) for f in listdir('experiments/')\
    if isfile(join('experiments/', f)) and f.find(exp_ext)==len(f)-5]
printt("List of experiment files: " + '\n'.join('{}: {}'.format(*k) for k in enumerate(json_files)), debug=True)


class Experiments(object):
    def __init__(self):
        json_dict = {}
        json_dicts = {}
        json_locations = {}
        for json_file in json_files:
            with open(json_file, 'r') as f:
                this_dict = json.load(f)
                json_dict = {**this_dict, **json_dict}
                json_dicts[json_file] = this_dict
                this_json_locations = {k: json_file for k in this_dict.keys()}
                json_locations = {**json_locations, **this_json_locations}
        self.experiments = json_dict
        self.experiments_json_files = json_dicts
        self.json_locations = json_locations
    def keys(self):
        '''
        Returns list of experiments
        '''
        return self.experiments.keys()
    def __getitem__(self, attr):
        if not(attr in self.experiments.keys()):
            printt("Experiment doesn't exist (retrieval)", error=True, stop=True)
        return self.experiments[attr]
    def __setitem__(self, attr, val):
        '''
        Let's use __setitem__ for updating the folder name of an experiment
        '''
        if not(attr in self.experiments.keys()):
            printt("Experiment doesn't exist (setting)", error=True, stop=True)
        experiment_id = attr
        folder_name = val
        self.save(experiment_id, folder_name)
    def update_experiment(self, experiment_id, attr, val):
        file_location = self.json_locations[experiment_id]
        self.experiments[experiment_id][attr] = val
        self.experiments_json_files[file_location][experiment_id][attr] = val
        self.update_json_file(file_location)
    def save(self, experiment_id, folder_name):
        file_location = self.json_locations[experiment_id]
        self.experiments[experiment_id]["save_folder_name"] = folder_name
        self.experiments_json_files[file_location][experiment_id]["save_folder_name"] = folder_name
        self.update_json_file(file_location)
    def update_json_file(self, json_file):
        this_dict = self.experiments_json_files[json_file]
        with open(json_file, 'w') as f:
            json.dump(this_dict, f, indent=4)
    def reset(self, experiment_id):
        '''
        Set the save_folder for an experiment to None
        '''
        file_location = self.json_locations[experiment_id]
        self.experiments[experiment_id]["save_folder_name"] = None
        self.experiments_json_files[file_location][experiment_id]["save_folder_name"] = None
        self.update_json_file(file_location)
