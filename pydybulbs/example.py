import numpy as np
from stokes import *
from analysis import *

# Load
load = [0, 0, 1000]
# Receiver
R = 5
f = 15*np.pi/180
t = 0
coord_sph = [R, f, t]
coord_cart = R * [np.sin(f), 0, np.cos(f)]
# Soil
Vsel = 150
dens = 2
nu = 0.30
ksi = 0.02
Gel = Vsel**2 * dens
Gvisc = Gel * (1+2*1j*ksi)
Vs = Vsel/np.sqrt(Gvisc/Gel)
# Frequency
omega = 3
omega_s = omega*R/Vs

[G,dGdx] = stokes_spherical(coordinates=coord_sph, shear=Gvisc, poisson=nu, omega_s=omega_s)
[u,dudx] = displacement(P=load, G=G, dGdx=dGdx)
e = strain_spherical(dudx=dudx, u=u, coordinates=coord_sph)
s = stress(strain=e, shear=Gvisc, poisson=nu)

[G,dGdx] = stokes_cartesian(coordinates=coord_cart, shear=Gvisc, poisson=nu, omega_s=omega_s)
[u,dudx] = displacement(P=load, G=G, dGdx=dGdx)
e = strain_cartesian(dudx=dudx)
s = stress(strain=e, shear=Gvisc, poisson=nu)

print('DONE')