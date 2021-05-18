import sys
import os

CURRPATH, TAIL = os.path.split(os.getcwd())
while CURRPATH != "/":
    if TAIL == 'p2d_fast_solver':
        if os.path.join(CURRPATH, TAIL) not in sys.path:
            sys.path.append(os.path.join(CURRPATH, TAIL))
        break
    CURRPATH, TAIL = os.path.split(CURRPATH)

from utils.derivatives import partials, compute_jac
from utils.unpack import unpack_fast, unpack