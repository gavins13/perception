from functools import reduce
import operator
import os
import traceback
import sys
from datetime import datetime
import json

perception_path =os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../")

with open(os.path.join(
    perception_path, "Config.perception"), "r") as config_file:
        Config = json.load(config_file)

def detect_cmd_arg(arg, retrieve_val=True, val_dtype=str, false_val=False):
    '''
    NOTE: Prefer this over argparse package since custom 'Perception' functionality will be added later

    This will look for the argument 'arg' in the Python command line input and if the retrieve_val is set to True, it will make sure to include an '=' sign in the input and retrieve the corresponding value

    If val_dtype is set, it will also perform a conversion on the argument to the specified data type
    '''
    try:
        assert isinstance(arg, str)
    except:
        raise ValueError("argument should be a string")
    arg = "--" + arg if arg.find("--") != 0 else arg
    arg = arg + "=" if retrieve_val is True else arg
    for i in range(len(sys.argv)):
        this_arg = sys.argv[i]
        if arg in this_arg:
            printt("Use a command-line given " + arg, info=True)
            if retrieve_val is True:
                val = this_arg.split(arg)
                val = val_dtype(val[1])
                printt(arg + " CMD ARG DETECTED: {}".format(val), info=True)
                return val
            else:
                return True
    '''if retrieve_val is True:
        return false_val
    else:
        return False'''
    return false_val



config_error_level = Config["debug_level"]
config_error_level = detect_cmd_arg("debug_level",
    retrieve_val=True, val_dtype=int, false_val=config_error_level)
def printt(val, warning=False, error=False, execution=False, info=False,
    stop=False, debug=False, full_file_path=None):
        '''
        WARNING, ERROR, EXECUTION, INFO
        '''
        os_error_level = os.environ.get('PYTHON_PERCEPTION_DEBUG_LEVEL')
        if os_error_level is not None:
            error_level = int(os_error_level)
        else:
            error_level = config_error_level
        for_print = None
        if (error is True) and (error_level > 0):
            for_print = "ERROR: {0}".format(val)
        elif (warning is True) and (error_level > 1):
            for_print = "WARNING: {0}".format(val)
        elif (info is True) and (error_level > 2):
            for_print = "INFO: {0}".format(val)
        elif (debug is True) and (error_level > 3):
            for_print = "DEBUG: {0}".format(val)
        elif execution is True:
            for_print = "EXECUTION: {0}".format(val)
        elif error_level > 4:
            for_print = "STREAM: {0}".format(val)

        if for_print is not None:
            print(for_print)
        if full_file_path is not None:
            with open(full_file_path, 'a') as f:
                for_print = "{0}".format(val) if\
                    for_print is None else for_print
                f.write(for_print + "\r\n")

        if stop is True:
            print("Stopped execution.")
            traceback.print_last()
            raise SystemExit

def path(fullpath, needle="**__path__**"):
    fullpath = fullpath.replace(needle, perception_path)
    return fullpath


flush_on_count = Config["log_flush_count"]
flush_on_count = detect_cmd_arg("log_flush_count",
    retrieve_val=True, val_dtype=int, false_val=flush_on_count)
flush_on_count = detect_cmd_arg("flush_count",
    retrieve_val=True, val_dtype=int, false_val=flush_on_count)
flush_on_count = detect_cmd_arg("lfc",
    retrieve_val=True, val_dtype=int, false_val=flush_on_count)
class Logger(object):
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "a")
        self.flush_on_count = flush_on_count # Number of lines before appending to file
        self.flush_counter = 0

    def write(self, message):
        self.terminal.write(message)
        #self.terminal.flush()
        null_messages = [";", "\r"]
        if message == "\n":
            self.log.write(message)
        else:
            datetimestr = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            datetimestr = datetimestr.replace(" ", "_")
            if message not in null_messages:
                self.log.write(datetimestr + ': ' + message + "\n")
            if self.flush_counter+1 == self.flush_on_count:
                self.log.flush()
                self.flush_counter = 0
            else:
                self.flush_counter += 1


    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass
