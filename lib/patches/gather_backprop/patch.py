import pathlib
import os
import shutil
from tensorflow.python.ops import array_grad
path = array_grad.__file__
current_path = str(pathlib.Path(__file__).parent.absolute())
patch_file = os.path.join(current_path, 'array_grad_fix.py')
print("Making Backup: Copying {} to {}".format(path, path+'_BACKUP'))
shutil.copy(path, path+'_BACKUP')

print("Applying Patch: Copying {} to {}".format(patch_file, path))
shutil.copy(patch_file, path)