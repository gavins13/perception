from functools import reduce
import operator
import os
import traceback
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
def printt(val, warning=False, error=False, execution=False, info=True, stop=False):
    '''
    WARNING, ERROR, EXECUTION, INFO
    '''
    conditions = []
    conditions.append((warning is True) and (error_level > 0))
    conditions.append((error is True) and (error_level > 1))
    conditions.append((execution is True))
    if reduce(operator.or, conditions) is True:
        print(val)

    if (warning is True) and (error_level > 0):
        print("WARNING: {0}".format(val))
    elif (error is True) and (error_level > 1):
        print("ERROR: {0}".format(val))
    elif (info is True) and (error_level > 2):
        print("INFO: {0}".format(val)))
    elif execution is True:
        print("EXECUTION: {0}".format(val))
    elif error_level > 3:
        print(val)

    if stop is True:
        print("Stopped execution.")
        traceback.print_tb()
        raise SystemExit