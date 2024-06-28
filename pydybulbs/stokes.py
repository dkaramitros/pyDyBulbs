from typing import List, Union
import numpy as np

def stokes_spherical(coordinates: Union[List[float],np.ndarray], shear: float=1, poisson: float=0.3, omega_s: float=1):
    """
    Computes Green's functions for the Stokes problem in spherical coordinates.

    Args:
        coordinates (Union[List[float],np.ndarray], optional): Receiver coordinates, defined as [radius R, aperture f, azimuth t]. Defaults to [1,1,1].
        shear (float, optional): Shear modulus G. Defaults to 1.
        poisson (float, optional): Poisson's ratio, n. Defaults to 0.3.
        omega_s (float, optional): Normalised frequency omega*R/Vs. Defaults to 1.

    Returns:
        list: Green's functions (3x3) for the frequency domain response in direction i due to a unit load in direction j,
            and derivatives (3x3x3) with respect to x(k).
    """
    # Rename variables
    R = coordinates[0]
    f = coordinates[1]
    t = coordinates[2]
    m = shear
    n = poisson
    Ws = omega_s
    # Compute normalised components
    [Y,X,dYdR,dXdR] = stokes(omega_s=Ws, poisson=n, radius=R)
    # Green's functions (Eqn 4.10):
    G = np.zeros([3,3], dtype = 'complex_')
    G[0,0] = ((Y+X)/(4*np.pi*m*R)) * np.sin(f)*np.cos(t) #a
    G[1,0] = (Y/(4*np.pi*m*R)) * np.cos(f)*np.cos(t) #b
    G[2,0] = (Y/(4*np.pi*m*R)) * (-np.sin(t)) #c
    G[0,1] = ((Y+X)/(4*np.pi*m*R)) * np.sin(f)*np.sin(t) #d
    G[1,1] = (Y/(4*np.pi*m*R)) * np.cos(f)*np.sin(t) #e
    G[2,1] = (Y/(4*np.pi*m*R)) * np.cos(t) #f
    G[0,2] = ((Y+X)/(4*np.pi*m*R)) * np.cos(f) #g
    G[1,2] = (Y/(4*np.pi*m*R)) * (-np.sin(f)) #h
    G[2,2] = 0 #i
    # Derivatives of Green's functions:
    dGdx = np.zeros([3,3,3], dtype = 'complex_')
    dGdx[0,0,0] = (1/(4*np.pi*m*R)) * (dYdR+dXdR-Y/R-X/R) * np.sin(f)*np.cos(t)
    dGdx[1,0,0] = (1/(4*np.pi*m*R)) * (dYdR-Y/R) * np.cos(f)*np.cos(t)
    dGdx[2,0,0] = (1/(4*np.pi*m*R)) * (dYdR-Y/R) * (-np.sin(t))
    dGdx[0,1,0] = (1/(4*np.pi*m*R)) * (dYdR+dXdR-Y/R-X/R) * np.sin(f)*np.sin(t)
    dGdx[1,1,0] = (1/(4*np.pi*m*R)) * (dYdR-Y/R) * np.cos(f)*np.sin(t)
    dGdx[2,1,0] = (1/(4*np.pi*m*R)) * (dYdR-Y/R) * np.cos(t)
    dGdx[0,2,0] = (1/(4*np.pi*m*R)) * (dYdR+dXdR-Y/R-X/R) * np.cos(f)
    dGdx[1,2,0] = (1/(4*np.pi*m*R)) * (dYdR-Y/R) * (-np.sin(f))
    dGdx[2,2,0] = 0
    dGdx[0,0,1] = ((Y+X)/(4*np.pi*m*R)) * np.cos(f)*np.cos(t)
    dGdx[1,0,1] = (Y/(4*np.pi*m*R)) * (-np.sin(f))*np.cos(t)
    dGdx[2,0,1] = 0
    dGdx[0,1,1] = ((Y+X)/(4*np.pi*m*R)) * np.cos(f)*np.sin(t)
    dGdx[1,1,1] = (Y/(4*np.pi*m*R)) * (-np.sin(f))*np.sin(t)
    dGdx[2,1,1] = 0
    dGdx[0,2,1] = ((Y+X)/(4*np.pi*m*R)) * (-np.sin(f))
    dGdx[1,2,1] = (Y/(4*np.pi*m*R)) * (-np.cos(f))
    dGdx[2,2,1] = 0
    dGdx[0,0,2] = ((Y+X)/(4*np.pi*m*R)) * np.sin(f)*(-np.sin(t))
    dGdx[1,0,2] = (Y/(4*np.pi*m*R)) * np.cos(f)*(-np.sin(t))
    dGdx[2,0,2] = (Y/(4*np.pi*m*R)) * (-np.cos(t))
    dGdx[0,1,2] = ((Y+X)/(4*np.pi*m*R)) * np.sin(f)*np.cos(t)
    dGdx[1,1,2] = (Y/(4*np.pi*m*R)) * np.cos(f)*np.cos(t)
    dGdx[2,1,2] = (Y/(4*np.pi*m*R)) * (-np.sin(t))
    dGdx[0,2,2] = 0
    dGdx[1,2,2] = 0
    dGdx[2,2,2] = 0
    # Return results
    return [G,dGdx]


def stokes_cartesian(coordinates: Union[List[float],np.ndarray], shear: float=1, poisson: float=0.3, omega_s: float=1):
    """
    Computes Green's functions for the Stokes problem in cartesian coordinates.

    Args:
        coordinates (Union[List[float],np.ndarray], optional): Receiver coordinates, defined as [x, y, z]. Defaults to [1,1,1].
        shear (float, optional): Shear modulus G. Defaults to 1.
        poisson (float, optional): Poisson's ratio, n. Defaults to 0.3.
        omega_s (float, optional): Normalised frequency omega*R/Vs. Defaults to 1.

    Returns:
        list: Green's functions (3x3) for the frequency domain response in direction i due to a unit load in direction j,
            and derivatives (3x3x3) with respect to x(k).
    """
    # Rename variables
    x = np.asarray(coordinates)
    m = shear
    n = poisson
    Ws = omega_s
    # Source-receiver distance
    R = np.linalg.norm(x)
    # Direction cosines
    g = x/R
    # Kronecker's delta
    d = np.identity(3)
    # Compute normalised components
    [Y,X,dYdR,dXdR] = stokes(omega_s=Ws, poisson=n, radius=R)
    # Green's functions (Eqn 4.4):
    G = np.zeros([3,3], dtype = 'complex_')
    for i in range(3):
        for j in range(3):
            G[i,j] = (1/(4*np.pi*m*R)) * (Y*d[i,j] + X*g[i]*g[j])
    # Green's functions derivatives (Eqn 4.5)
    dGdx = np.zeros([3,3,3], dtype = 'complex_')
    for i in range(3):
        for j in range(3):
            for k in range(3):
                dGdx[i,j,k] = (1/(4*np.pi*m*R)) * (g[k] * ((dYdR-Y/R)*d[i,j] + (dXdR-3*X/R)*g[i]*g[j]) + (X/R) * (d[i,k]*g[j] + d[j,k]*g[i]))

    return [G,dGdx]


def stokes(omega_s: float=1, poisson: float=0.3, radius: float=1):
    """
    Computes dimensionless components of Green's functions for Stoke's problem.

    Args:
        omega_s (float, optional): Normalised frequency (omega*R/Vs). Defaults to 1.
        poisson (float, optional): Poisson's ratio. Defaults to 0.3.

    Returns:
        List: Y, X, dYdR, dXdR
    """
    # Rename variables
    Ws = omega_s
    n = poisson
    R = radius
    # P/S-wave velocity ratio (Eqn 4.2)
    a = np.sqrt((1-2*n)/(2*(1-n)))
    Wp = a*Ws
    # Normalised components and derivatives (Eqns 4.6 to 4.9)
    Y = np.exp(-1j*Wp)*a**2*(1j/Wp+1/Wp**2) + np.exp(-1j*Ws)*(1-1j/Ws-1./Ws**2)
    X = np.exp(-1j*Wp)*a**2*(1-3*1j/Wp-3./Wp**2) - np.exp(-1j*Ws)*(1-3*1j/Ws-3./Ws**2)
    dYdR = (1/R)*(np.exp(-1j*Wp)*a**2*(1-2*1j/Wp-2/Wp**2) - np.exp(-1j*Ws)*(1+1j*Ws-2*1j/Ws-2/Ws**2))
    dXdR = (1/R)*(np.exp(-1j*Ws)*(3+1j*Ws-6*1j/Ws-6/Ws**2) - np.exp(-1j*Wp)*a**2*(3+1j*Wp-6*1j/Wp-6/Wp**2))
    # Return results
    return [Y,X,dYdR,dXdR]