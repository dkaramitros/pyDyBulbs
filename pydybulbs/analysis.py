from typing import List, Union
import numpy as np


def displacement(P: Union[List[float],np.ndarray], G: np.ndarray, dGdx: np.ndarray):
    """
    Computes displacement and displacement gradients.

    Args:
        G (np.ndarray): Green's function for the frequency domain response in direction i
            due to a unit load in direction j (defined in cartesian coordinates x,y,z).
        dGdx (np.ndarray): Derivative of Green's functions with respect to x[k].
        P (Union[List[float],np.ndarray], optional): Load vector in cartesian coordinates x,y,z. Defaults to [0,0,1].

    Returns:
        list: displacements vector (3x1), displacement gradients (3x3).
    """
    # Displacement Vector (3x1) - Using G[i,j]*P[j]
    u = np.zeros([3], dtype = 'complex_')
    for i in range(3):
        for j in range(3):
            u[i] += G[i,j]*P[j]
    # Displacement Gradient (3x3) - Using dG[i,j]*P[j]/dx[k]
    dudx = np.zeros([3,3], dtype = 'complex_')
    for i in range(3):
        for k in range(3):
            for j in range(3):
                dudx[i,k] += dGdx[i,j,k]*P[j]
    # Return results
    return [u,dudx]


def strain_spherical(dudx: np.ndarray, u: np.ndarray, coordinates: Union[List[float],np.ndarray]):
    """
    Computes strains from displacement gradients.
    Strains are given in geotechnical sign convention (compression positive).

    Args:
        u (np.ndarray): Displacements.
        dudx (np.ndarray): Displacement gradients.
        coordinates (Union[List[float],np.ndarray]): _description_

    Returns:
        np.ndarray: Strain vector (6x1) [x,y,z,yz,zx,xy].
    """
    # Rename variables
    R = coordinates[0]
    f = coordinates[1]
    # Strain Vector (Eqns 1.85)
    e = np.zeros(6, dtype = 'complex_')
    e[0] = dudx[0,0] #R
    e[1] = u[0]/R+dudx[1,1]/R #f (phi)
    e[2] = u[0]/R+(1/np.tan(f))*u[1]/R+(1/np.sin(f))*dudx[2,2]/R #t (theta)
    e[3] = (1/np.sin(f))*dudx[1,2]/R+dudx[2,1]/R-(1/np.tan(f))*u[2]/R #ft
    e[4] = (1/np.sin(f))*dudx[0,2]/R+dudx[2,0]/R-u[2]/R #Rt
    e[5] = dudx[0,1]/R+dudx[1,0]-u[1]/R #Rf
    # Switch to geotechnical notation
    e = -e
    # Return results
    return e


def strain_cartesian(dudx: np.ndarray):
    """
    Computes strains from displacement gradients.
    Strains are given in geotechnical sign convention (compression positive).

    Args:
        dudx (np.ndarray): Displacement gradients.

    Returns:
        np.ndarray: Strain vector (6x1) [x,y,z,yz,zx,xy].
    """
    # Strain Vector (Eqns 1.46 & 1.48)
    e = np.zeros(6, dtype = 'complex_')
    e[0] = dudx[0,0] #x
    e[1] = dudx[1,1] #y
    e[2] = dudx[2,2] #z
    e[3] = dudx[2,1]+dudx[1,2] #yz
    e[4] = dudx[0,2]+dudx[2,0] #zx
    e[5] = dudx[1,0]+dudx[0,1] #xy
    # Switch to Geotechnical Notation
    e = -e
    return e


def stress(strain: Union[List[float],np.ndarray], shear: float=1, poisson: float=0.3):
    """
    Computes strains from stresses.

    Args:
        strain (Union[List[float],np.ndarray]): Strain vector (6x1).
        shear (float, optional): Shear modulus. Defaults to 1.
        poisson (float, optional): Poisson's ratio. Defaults to 0.3.

    Returns:
        np.ndarray: Stress vector (6x1) [x,y,z,yz,zx,xy].
    """
    # Rename variables
    e = strain
    m = shear
    n = poisson
    # Lame constant
    l = 2*m*n/(1-2*n)
    # Volumetric strain
    evol = e[0]+e[1]+e[2]
    # Stress Vector (Eqns 1.53 & 1.57)
    s = np.zeros(6, dtype = 'complex_')
    s[0] = 2*m*e[0]+l*evol #x
    s[1] = 2*m*e[1]+l*evol #y
    s[2] = 2*m*e[2]+l*evol #z
    s[3] = m*e[3] #yz
    s[4] = m*e[4] #zx
    s[5] = m*e[5] #xy
    # Return results
    return s