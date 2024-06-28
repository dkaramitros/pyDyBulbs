import numpy as np
from stokes import *

res = greens_cartesian([1,1,1],1,0.3,2)

print(res)