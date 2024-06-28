import numpy as np
from stokes import *
from analysis import *

res = stokes_spherical([1,1,1],1,0.3,2)
an1 = analysis_spherical([1,1,1],res[0],res[1],1,0.3,1,0.3)

res = stokes_cartesian([1,1,1],1,0.3,2)
an2 = analysis_cartesian([1,1,1],res[0],res[1],1,0.3)

print(an1)
print(an2)