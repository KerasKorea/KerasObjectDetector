from __future__ import absolute_import
from __future__ import print_function
import sys 

# Default backend: Retinanet.
_BACKEND = 'retinanet'

# Import backend functions.
if _BACKEND == 'retinanet':
    sys.stderr.write('Using retinanet backend\n')
    from .retinanet_backend import *

else:
    ValueError('Unable to import backend : ' + str(_BACKEND))

def backend():
    return _BACKEND