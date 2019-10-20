from __future__ import absolute_import
from __future__ import print_function
import sys 

# Default backend: Retinanet.
_BACKEND = 'ssd'

# Import backend functions.
if _BACKEND == 'retinanet':
    sys.stderr.write('Using retinanet backend\n')
    from .retinanet_backend import *

elif _BACKEND == 'yolov3':
    sys.stderr.write('Using yolov3 backend\n')
    from .yolov3_backend import *

elif _BACKEND == 'ssd':
    sys.stderr.write('Using SSD backend\n')
    from .ssd_backend import *

else:
    ValueError('Unable to import backend : ' + str(_BACKEND))

def backend():
    return _BACKEND