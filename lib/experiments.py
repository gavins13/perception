import json
from os import listdir
from os.path import isfile, join
from .misc import printt
import os
import copy

path = os.path.join(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../"
        ),
        "experiments/")

exp_ext = '.json'
json_files = [join(path, f) for f in listdir(path)\
    if isfile(join(path, f)) and f.find(exp_ext)==len(f)-5]
printt("List of perception experiment files: " + '\n'.join('{}: {}'.format(*k) for k in enumerate(json_files)), debug=True)


class Experiments(object):
    def __init__(self, experiments_file=None):
        self.experiments_file = experiments_file
        json_dict = {}
        json_dicts = {}
        json_locations = {}
        all_json_files= json_files[:]
        if experiments_file is not None:
            all_json_files.append(experiments_file)
        if len(all_json_files) == 0:
            printt("No experiments file loaded", info=True)
        for json_file in all_json_files:
            with open(json_file, 'r') as f:
                this_dict = json.load(f)
                json_dict = {**this_dict, **json_dict}
                json_dicts[json_file] = this_dict
                this_json_locations = {k: json_file for k in this_dict.keys()}
                json_locations = {**json_locations, **this_json_locations}
        printt(all_json_files, debug=True)
        '''
        Structure: self.experiments[EXPERIMENT_ID] -> All attributes
        '''
        self.experiments = json_dict

        '''
        Structure: self.experiments_json_files[JSON_FILE_NAME][EXPERIMENT_ID]->All Attributes
        '''
        self.experiments_json_files = json_dicts

        '''
        Structure: self.json_locations[EXPERIMENT_ID]->JSON FILE LOCATION
        '''
        self.json_locations = json_locations

        '''
        '''
        self.experiments_w_inheritance = copy.deepcopy(self.experiments)
        self.inheritance()

    def inherit(self, inherited_exp, exp):
        '''
        The parent experiment `inherited_exp' will be inherited to `exp' and
        exp will rewrite properties of `inherited_exp' in a nested fashion but
        not for list instances so please only use dictionaries!
        '''
        new_dict = copy.deepcopy(inherited_exp)
        for key, item in exp.items():
            if isinstance(item, dict) is True:
                item = self.inherit(inherited_exp[key], item)
            new_dict[key] = item
        return new_dict

    def inheritance(self):
        for experiment_id in self.experiments_w_inheritance.keys():
            if '__inherit__' in self.experiments_w_inheritance[experiment_id].keys():
                this_dict = copy.deepcopy(self.experiments_w_inheritance[experiment_id])
                to_inherit_name = this_dict['__inherit__']
                to_inherit = self.experiments_w_inheritance[to_inherit_name]
                del(this_dict['__inherit__'])
                new_dict = self.inherit(to_inherit, this_dict)
                self.experiments_w_inheritance[experiment_id] = new_dict

    def keys(self):
        '''
        Returns list of experiments
        '''
        return self.experiments.keys()
    def __getitem__(self, attr):
        if attr is None:
            return
        if not(attr in self.experiments.keys()):
            printt("Experiment doesn't exist (retrieval)", error=True, stop=True)
        return self.experiments_w_inheritance[attr]
    def __setitem__(self, attr, val):
        '''
        Let's use __setitem__ for updating the folder name of an experiment
        '''
        if attr is None:
            return
        if not(attr in self.experiments.keys()):
            printt("Experiment doesn't exist (setting)", error=True, stop=True)
        experiment_id = attr
        folder_name = val
        self.save(experiment_id, folder_name)
    def update_experiment(self, experiment_id, attr, val):
        if experiment_id is None:
            return
        else:
            self.__init__(experiments_file=self.experiments_file)
            file_location = self.json_locations[experiment_id]
            self.experiments[experiment_id][attr] = val
            self.experiments_json_files[file_location][experiment_id][attr] = val
            self.update_json_file(file_location)
    def save(self, experiment_id, folder_name):
        if experiment_id is None:
            return
        self.__init__(experiments_file=self.experiments_file)
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
        if experiment_id is None:
            return
        file_location = self.json_locations[experiment_id]
        self.experiments[experiment_id]["save_folder_name"] = None
        self.experiments_json_files[file_location][experiment_id]["save_folder_name"] = None
        self.update_json_file(file_location)
