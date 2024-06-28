import numpy as np
from stokes import *
from analysis import *

#res = greens_spherical([1,1,1],1,0.3,2)
#an = analysis_spherical([1,1,1],res[0],res[1],1,0.3,1,0.3)

res = greens_cartesian([1,1,1],1,0.3,2)
an = analysis_cartesian([1,1,1],res[0],res[1],1,0.3)

print(an)