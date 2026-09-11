"""
Compatibility shim for the notebooks under notebooks/, which were written
against Di Carlo's original module name (`from IsingRG import *`) while
this repository keeps the extended implementation under the more
descriptive name IsingRG_3spin.py. Import from IsingRG_3spin.py directly
in new code; this file exists only so the existing notebooks run
unmodified once src/ is on sys.path.
"""
from IsingRG_3spin import *
