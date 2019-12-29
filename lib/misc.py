from functools import reduce
import operator
import os
import traceback
import sys
'''
from experiment import debug_level
debug_level:
0 = No printing except execution
1 = Warnings
2 = Errors
3 = Information
4 = All (Full debugging)
'''

error_level = 4
def printt(val, warning=False, error=False, execution=False, info=False, stop=False, debug=False):
    '''
    WARNING, ERROR, EXECUTION, INFO
    '''
    if (warning is True) and (error_level > 0):
        print("WARNING: {0}".format(val))
    elif (error is True) and (error_level > 1):
        print("ERROR: {0}".format(val))
    elif (info is True) and (error_level > 2):
        print("INFO: {0}".format(val))
    elif (debug is True) and (error_level > 3):
        print("DEBUG: {0}".format(val))
    elif execution is True:
        print("EXECUTION: {0}".format(val))
    elif error_level > 4:
        print(val)

    if stop is True:
        print("Stopped execution.")
        traceback.print_exc()
        raise SystemExit




def detect_cmd_arg(arg, retrieve_val=True, val_dtype=str, false_val=None):
    '''
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
                printt(arg + " CMD ARG DETECTED", info=True)
                return val
            else:
                return True
    if retrieve_val is True:
        return false_val
    else:
        return False
