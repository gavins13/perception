import os
import pathlib
from subprocess import run

current_path = str(pathlib.Path(__file__).parent.absolute())

patches = os.path.join(current_path, 'lib/patches/')


patch = os.path.join(patches, 'gather_backprop/patch.py')
run(['python', patch])

