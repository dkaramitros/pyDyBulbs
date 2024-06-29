from typing import List, Union
import numpy as np

def stokes_spherical(coordinates: Union[List[float], np.ndarray], shear: float=1, poisson: float=0.3, omega_s: float=1):
    """
    Computes Green's functions for the Stokes problem in spherical coordinates.

    Args:
        coordinates (Union[List[float], np.ndarray], optional): Receiver coordinates, defined as [radius R, aperture f, azimuth t]. Defaults to [1, 1, 1].
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
    [Y, X, dYdR, dXdR] = stokes(omega_s=Ws, poisson=n, radius=R)    
    # Green's functions (Eqn 4.10):
    G = np.zeros([3, 3], dtype='complex_')
    G[0, 0] = ((Y + X) / (4 * np.pi * m * R)) * np.sin(f) * np.cos(t)  # a
    G[1, 0] = (Y / (4 * np.pi * m * R)) * np.cos(f) * np.cos(t)        # b
    G[2, 0] = (Y / (4 * np.pi * m * R)) * (-np.sin(t))                 # c
    G[0, 1] = ((Y + X) / (4 * np.pi * m * R)) * np.sin(f) * np.sin(t)  # d
    G[1, 1] = (Y / (4 * np.pi * m * R)) * np.cos(f) * np.sin(t)        # e
    G[2, 1] = (Y / (4 * np.pi * m * R)) * np.cos(t)                    # f
    G[0, 2] = ((Y + X) / (4 * np.pi * m * R)) * np.cos(f)              # g
    G[1, 2] = (Y / (4 * np.pi * m * R)) * (-np.sin(f))                 # h
    G[2, 2] = 0                                                        # i
    # Derivatives of Green's functions:
    dGdx = np.zeros([3, 3, 3], dtype='complex_')
    dGdx[0, 0, 0] = (1 / (4 * np.pi * m * R)) * (dYdR + dXdR - Y / R - X / R) * np.sin(f) * np.cos(t)
    dGdx[1, 0, 0] = (1 / (4 * np.pi * m * R)) * (dYdR - Y / R) * np.cos(f) * np.cos(t)
    dGdx[2, 0, 0] = (1 / (4 * np.pi * m * R)) * (dYdR - Y / R) * (-np.sin(t))
    dGdx[0, 1, 0] = (1 / (4 * np.pi * m * R)) * (dYdR + dXdR - Y / R - X / R) * np.sin(f) * np.sin(t)
    dGdx[1, 1, 0] = (1 / (4 * np.pi * m * R)) * (dYdR - Y / R) * np.cos(f) * np.sin(t)
    dGdx[2, 1, 0] = (1 / (4 * np.pi * m * R)) * (dYdR - Y / R) * np.cos(t)
    dGdx[0, 2, 0] = (1 / (4 * np.pi * m * R)) * (dYdR + dXdR - Y / R - X / R) * np.cos(f)
    dGdx[1, 2, 0] = (1 / (4 * np.pi * m * R)) * (dYdR - Y / R) * (-np.sin(f))
    dGdx[2, 2, 0] = 0
    dGdx[0, 0, 1] = ((Y + X) / (4 * np.pi * m * R)) * np.cos(f) * np.cos(t)
    dGdx[1, 0, 1] = (Y / (4 * np.pi * m * R)) * (-np.sin(f)) * np.cos(t)
    dGdx[2, 0, 1] = 0
    dGdx[0, 1, 1] = ((Y + X) / (4 * np.pi * m * R)) * np.cos(f) * np.sin(t)
    dGdx[1, 1, 1] = (Y / (4 * np.pi * m * R)) * (-np.sin(f)) * np.sin(t)
    dGdx[2, 1, 1] = 0
    dGdx[0, 2, 1] = ((Y + X) / (4 * np.pi * m * R)) * (-np.sin(f))
    dGdx[1, 2, 1] = (Y / (4 * np.pi * m * R)) * (-np.cos(f))
    dGdx[2, 2, 1] = 0
    dGdx[0, 0, 2] = ((Y + X) / (4 * np.pi * m * R)) * np.sin(f) * (-np.sin(t))
    dGdx[1, 0, 2] = (Y / (4 * np.pi * m * R)) * np.cos(f) * (-np.sin(t))
    dGdx[2, 0, 2] = (Y / (4 * np.pi * m * R)) * (-np.cos(t))
    dGdx[0, 1, 2] = ((Y + X) / (4 * np.pi * m * R)) * np.sin(f) * np.cos(t)
    dGdx[1, 1, 2] = (Y / (4 * np.pi * m * R)) * np.cos(f) * np.cos(t)
    dGdx[2, 1, 2] = (Y / (4 * np.pi * m * R)) * (-np.sin(t))
    dGdx[0, 2, 2] = 0
    dGdx[1, 2, 2] = 0
    dGdx[2, 2, 2] = 0
    return [G, dGdx]


def stokes_cartesian(coordinates: Union[List[float], np.ndarray], shear: float=1, poisson: float=0.3, omega_s: float=1):
    """
    Computes Green's functions for the Stokes problem in cartesian coordinates.

    Args:
        coordinates (Union[List[float], np.ndarray], optional): Receiver coordinates, defined as [x, y, z]. Defaults to [1, 1, 1].
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
    g = x / R
    # Kronecker's delta
    d = np.identity(3)
    # Compute normalised components
    [Y, X, dYdR, dXdR] = stokes(omega_s=Ws, poisson=n, radius=R)
    # Green's functions (Eqn 4.4):
    G = np.zeros([3, 3], dtype='complex_')
    for i in range(3):
        for j in range(3):
            G[i, j] = (1 / (4 * np.pi * m * R)) * (Y * d[i, j] + X * g[i] * g[j])
    # Green's functions derivatives (Eqn 4.5)
    dGdx = np.zeros([3, 3, 3], dtype='complex_')
    for i in range(3):
        for j in range(3):
            for k in range(3):
                dGdx[i, j, k] = (1 / (4 * np.pi * m * R)) * (
                        g[k] * ((dYdR - Y / R) * d[i, j] + (dXdR - 3 * X / R) * g[i] * g[j]) +
                        (X / R) * (d[i, k] * g[j] + d[j, k] * g[i]))
    return [G, dGdx]


def stokes(omega_s: float=1, poisson: float=0.3, radius: float=1):
    """
    Computes dimensionless components of Green's functions for Stoke's problem
    (used for both spherical and cartesian coordinates).

    Args:
        omega_s (float, optional): Normalised frequency (omega*R/Vs). Defaults to 1.
        poisson (float, optional): Poisson's ratio. Defaults to 0.3.
        radius (float, optional): Radius. Defaults to 1.

    Returns:
        List: Y, X, dYdR, dXdR
    """
    # Rename variables
    Ws = omega_s
    n = poisson
    R = radius
    # P/S-wave velocity ratio (Eqn 4.2)
    a = np.sqrt((1 - 2 * n) / (2 * (1 - n)))
    Wp = a * Ws
    # Normalised components and derivatives (Eqns 4.6 to 4.9)
    if Ws == 0:
        Y = (2 * (1 - 2 * n) + 1) / (4 * (1 - n))
        X = 1 / (4 * (1 - n))
        dYdR = 0
        dXdR = 0
    else:
        Y = np.exp(-1j * Wp) * a**2 * (1j / Wp + 1 / Wp**2) + np.exp(-1j * Ws) * (1 - 1j / Ws - 1. / Ws**2)
        X = np.exp(-1j * Wp) * a**2 * (1 - 3 * 1j / Wp - 3. / Wp**2) - np.exp(-1j * Ws) * (1 - 3 * 1j / Ws - 3. / Ws**2)
        dYdR = (1 / R) * (np.exp(-1j * Wp) * a**2 * (1 - 2 * 1j / Wp - 2 / Wp**2) - np.exp(-1j * Ws) * (1 + 1j * Ws - 2 * 1j / Ws - 2 / Ws**2))
        dXdR = (1 / R) * (np.exp(-1j * Ws) * (3 + 1j * Ws - 6 * 1j / Ws - 6 / Ws**2) - np.exp(-1j * Wp) * a**2 * (3 + 1j * Wp - 6 * 1j / Wp - 6 / Wp**2))
    return [Y, X, dYdR, dXdR]


def displacement(dP: Union[List[float], np.ndarray], G: np.ndarray, dGdx: np.ndarray):
    """
    Computes displacement and displacement gradients (both for spherical and cartesian coordinates).

    Args:
        G (np.ndarray): Green's function for the frequency domain response in direction i
            due to a unit load in direction j (defined in cartesian coordinates x,y,z).
        dGdx (np.ndarray): Derivative of Green's functions with respect to x[k].
        dP (Union[List[float], np.ndarray]): Load vector 2*P in cartesian coordinates x,y,z.

    Returns:
        tuple: displacements vector (3x1), displacement gradients (3x3).
    """
    # Ensure P is a NumPy array
    P = np.array(dP, dtype='complex_')
    # Displacement Vector (3x1) - Using G[i,j]*P[j]
    u = np.dot(G, P)
    # Displacement Gradient (3x3) - Using dG[i,j,k]*P[j]
    dudx = np.einsum('ijk,j->ik', dGdx, P)
    return u, dudx


def strain_spherical(dudx: np.ndarray, u: np.ndarray, coordinates: Union[List[float], np.ndarray]):
    """
    Computes strains from displacement gradients (spherical coordinates only).
    Strains are given in geotechnical sign convention (compression positive).

    Args:
        u (np.ndarray): Displacements.
        dudx (np.ndarray): Displacement gradients.
        coordinates (Union[List[float], np.ndarray]): Spherical coordinates [R, phi, theta].

    Returns:
        np.ndarray: Strain vector (6x1) [R, phi, theta, phi-theta, R-theta, R-phi].
    """
    R = coordinates[0]
    phi = coordinates[1]
    # Strain Vector (Eqns 1.85)
    e = np.zeros(6, dtype='complex_')
    e[0] = dudx[0, 0]  # R
    e[1] = u[0] / R + dudx[1, 1] / R  # phi
    e[2] = u[0] / R + (1 / np.tan(phi)) * u[1] / R + (1 / np.sin(phi)) * dudx[2, 2] / R  # theta
    e[3] = (1 / np.sin(phi)) * dudx[1, 2] / R + dudx[2, 1] / R - (1 / np.tan(phi)) * u[2] / R  # phi-theta
    e[4] = (1 / np.sin(phi)) * dudx[0, 2] / R + dudx[2, 0] / R - u[2] / R  # R-theta
    e[5] = dudx[0, 1] / R + dudx[1, 0] - u[1] / R  # R-phi
    # Switch to geotechnical notation
    e = -e
    return e


def strain_cartesian(dudx: np.ndarray):
    """
    Computes strains from displacement gradients (cartesian coordinates only).
    Strains are given in geotechnical sign convention (compression positive).

    Args:
        dudx (np.ndarray): Displacement gradients.

    Returns:
        np.ndarray: Strain vector (6x1) [x,y,z,yz,zx,xy].
    """
    # Strain Vector (Eqns 1.46 & 1.48)
    e = np.zeros(6, dtype='complex_')
    e[0] = dudx[0, 0]  # x
    e[1] = dudx[1, 1]  # y
    e[2] = dudx[2, 2]  # z
    e[3] = dudx[2, 1] + dudx[1, 2]  # yz
    e[4] = dudx[0, 2] + dudx[2, 0]  # zx
    e[5] = dudx[1, 0] + dudx[0, 1]  # xy
    # Switch to geotechnical notation
    e = -e
    return e


def stress(strain: Union[List[float], np.ndarray], shear: float = 1, poisson: float = 0.3):
    """
    Computes stresses from strains (both for spherical and cartesian coordinates).

    Args:
        strain (Union[List[float], np.ndarray]): Strain vector (6x1).
        shear (float, optional): Shear modulus. Defaults to 1.
        poisson (float, optional): Poisson's ratio. Defaults to 0.3.

    Returns:
        np.ndarray: Stress vector (6x1) [x,y,z,yz,zx,xy].
    """
    # Rename variables
    e = np.array(strain, dtype='complex_')
    m = shear
    n = poisson
    # Lame constant
    l = 2 * m * n / (1 - 2 * n)
    # Volumetric strain
    evol = e[0] + e[1] + e[2]
    # Stress Vector (Eqns 1.53 & 1.57)
    s = np.zeros(6, dtype='complex_')
    s[0] = 2 * m * e[0] + l * evol  # x
    s[1] = 2 * m * e[1] + l * evol  # y
    s[2] = 2 * m * e[2] + l * evol  # z
    s[3] = m * e[3]  # yz
    s[4] = m * e[4]  # zx
    s[5] = m * e[5]  # xy
    return s