import numpy as np

def greens_spherical(x,m,n,Ws):
    # Green's Functions for Stokes Problem in Spherical Coordinates
    # See Chapter 4.2 in Kausel's book on Elastodynamics
    #
    # Symbol/Variable Definitions
    # (descriptions in parentheses refer to the symbols used by Kausel)
    #
    # Input:
    # - x(3): Spherical coordinates (receiver) - R, f(phi), t(theta)
    # - m (greek): Shear modulus
    # - n (greek): Poisson's ratio
    # - Ws (greek): Dimensionless frequency for S (shear) waves
    #
    # Output:
    # - G(3,3) (small): Green�s function for the frequency domain response in
    #                   direction i (defined in spherical coordinates R,f,t)
    #                   due to a unit load in direction j (defined in cartesian
    #                   coordinates x,y,z) 
    # - dGdx(3,3,3): derivative with respect to x(k) (defined in spherical
    #                coordinates R,f,t)
    #
    # Intermediate:
    # - a: Ratio of S- and P-wave velocities
    # - Wp: Dimensionless frequency for P (dilatational) waves
    # - g(3) (greek): Direction cosine of R with ith axis
    # - X (small/greek): Dimensionless component function of Green�s functions
    # - Y (small/greek): Dimensionless component function of Green�s functions
    # - dXdR: Derivative of X with respect to R
    # - dYdR: Derivative of Y with respect to R

    # Source-receiver distance (in 3-D space)
    # Convert to array
    x = np.asarray(x)
    R = x[0]
    f = x[1]
    t = x[2]

    # Eqn 4.2:
    a = np.sqrt((1-2*n)/(2*(1-n)))
    Wp = a*Ws
    # Eqn 4.3:
    g = x/R
    # Eqn 4.6:
    Y = np.exp(-1j*Wp)*a**2*(1j/Wp+1/Wp**2) + np.exp(-1j*Ws)*(1-1j/Ws-1./Ws**2)
    # Eqn 4.7:
    X = np.exp(-1j*Wp)*a**2*(1-3*1j/Wp-3./Wp**2) - np.exp(-1j*Ws)*(1-3*1j/Ws-3./Ws**2)
    # Eqn 4.8:
    dYdR = (1/R)*(np.exp(-1j*Wp)*a**2*(1-2*1j/Wp-2/Wp**2) - np.exp(-1j*Ws)*(1+1j*Ws-2*1j/Ws-2/Ws**2))
    # Eqn 4.9:
    dXdR = (1/R)*(np.exp(-1j*Ws)*(3+1j*Ws-6*1j/Ws-6/Ws**2) - np.exp(-1j*Wp)*a**2*(3+1j*Wp-6*1j/Wp-6/Wp**2))

    # Eqn 4.10:
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

    return [G,dGdx]


def greens_cartesian(x,m,n,Ws):
    # Green's Functions for Stokes Problem in Cartesian Coordinates
    # See Chapter 4.2 in Kausel's book on Elastodynamics
    #
    # Symbol/Variable Definitions
    # (descriptions in parentheses refer to the symbols used by Kausel)
    # 
    # Input:
    # - x(3): Cartesian coordinates (receiver)
    # - m (greek): Shear modulus
    # - n (greek): Poisson's ratio
    # - Ws (greek): Dimensionless frequency for S (shear) waves
    #
    # Output:
    # - G(3,3) (small): Green�s function for the frequency domain response in
    #                   direction i due to a unit load in direction j
    # - dGdx(3,3,3): derivative with respect to x(k)
    #
    # Intermediate:
    # - d(3,3) (greek): Kronecker delta
    # - R: Source�receiver distance in 3-D space
    # - a: Ratio of S- and P-wave velocities
    # - Wp: Dimensionless frequency for P (dilatational) waves
    # - g(3) (greek): Direction cosine of R with ith axis
    # - X (small/greek): Dimensionless component function of Green�s functions
    # - Y (small/greek): Dimensionless component function of Green�s functions
    # - dXdR: Derivative of X with respect to R
    # - dYdR: Derivative of Y with respect to R

    # Basics
    d = np.identity(3)
    # Eqn 4.1:
    x = np.asarray(x)
    R = np.linalg.norm(x)
    # Eqn 4.2:
    a = np.sqrt((1-2*n)/(2*(1-n)))
    Wp = a*Ws
    # Eqn 4.3:
    g = x/R
    # Eqn 4.6:
    Y = np.exp(-1j*Wp)*a**2*(1j/Wp+1./Wp**2) + np.exp(-1j*Ws)*(1-1j/Ws-1/Ws**2)
    # Eqn 4.7:
    X = np.exp(-1j*Wp)*a**2*(1-3*1j/Wp-3/Wp**2) - np.exp(-1j*Ws)*(1-3*1j/Ws-3/Ws**2)
    # Eqn 4.8:
    dYdR = (1/R)*(np.exp(-1j*Wp)*a**2*(1-2*1j/Wp-2/Wp**2)-np.exp(-1j*Ws)*(1+1j*Ws-2*1j/Ws-2/Ws**2))
    # Eqn 4.9:
    dXdR = (1/R)*(np.exp(-1j*Ws)*(3+1j*Ws-6*1j/Ws-6/Ws**2)-np.exp(-1j*Wp)*a**2*(3+1j*Wp-6*1j/Wp-6/Wp**2))
    # Eqn 4.4:
    G = np.zeros([3,3], dtype = 'complex_')
    for i in range(3):
        for j in range(3):
            G[i,j] = (1/(4*np.pi*m*R)) * (Y*d[i,j] + X*g[i]*g[j])
    # Eqn 4.5:
    dGdx = np.zeros([3,3,3], dtype = 'complex_')
    for i in range(3):
        for j in range(3):
            for k in range(3):
                dGdx[i,j,k] = (1/(4*np.pi*m*R)) * (g[k] * ((dYdR-Y/R)*d[i,j] + (dXdR-3*X/R)*g[i]*g[j]) + (X/R) * (d[i,k]*g[j] + d[j,k]*g[i]))

    return [G,dGdx]