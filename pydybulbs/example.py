import numpy as np
from stokes import *

# Load
load = [0, 0, 2000]
# Receiver
R = 5
f = 15*np.pi/180
t = 0
coord_sph = [R, f, t]
coord_cart = [R*np.sin(f), 0, R*np.cos(f)]
# Soil
Vsel = 150
dens = 2
nu = 0.30
ksi = 0
Gel = Vsel**2 * dens
Gvisc = Gel * (1+2*1j*ksi)
Vs = Vsel/np.sqrt(Gvisc/Gel)
# Frequency
omega = 3
omega_s = omega*R/Vs

# Spherical coordinates
[G,dGdx] = stokes_spherical(coordinates=coord_sph, shear=Gvisc, poisson=nu, omega_s=omega_s)
[u,dudx] = displacement(P=load, G=G, dGdx=dGdx)
e = strain_spherical(dudx=dudx, u=u, coordinates=coord_sph)
s = stress(strain=e, shear=Gvisc, poisson=nu)

print("\nAnalysis in spherical coordinates:")
print("Displacements:")
for i,x in enumerate(u):
    print(f"  {i+1}: Amplitude: {abs(x):.5f}, Phase angle: {np.degrees(np.angle(x)):.3f} degrees")
print("Strains:")
for i,x in enumerate(e):
    print(f"  {i+1}: Amplitude: {100*abs(x):.5f} %, Phase angle: {np.degrees(np.angle(x)):.3f} degrees")
print("Stresses:")
for i,x in enumerate(s):
    print(f"  {i+1}: Amplitude: {abs(x):.5f}, Phase angle: {np.degrees(np.angle(x)):.3f} degrees")

# Cartesian coordinates
[G,dGdx] = stokes_cartesian(coordinates=coord_cart, shear=Gvisc, poisson=nu, omega_s=omega_s)
[u,dudx] = displacement(P=load, G=G, dGdx=dGdx)
e = strain_cartesian(dudx=dudx)
s = stress(strain=e, shear=Gvisc, poisson=nu)

print("\nAnalysis in cartesian coordinates:")
print("Displacements:")
for i,x in enumerate(u):
    print(f"  {i+1}: Amplitude: {abs(x):.5f}, Phase angle: {np.degrees(np.angle(x)):.3f} degrees")
print("Strains:")
for i,x in enumerate(e):
    print(f"  {i+1}: Amplitude: {100*abs(x):.5f} %, Phase angle: {np.degrees(np.angle(x)):.3f} degrees")
print("Stresses:")
for i,x in enumerate(s):
    print(f"  {i+1}: Amplitude: {abs(x):.5f}, Phase angle: {np.degrees(np.angle(x)):.3f} degrees")