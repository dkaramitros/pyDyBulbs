import numpy as np


def analysis_spherical(P,G,dGdx,m,n,R,f):
    # Spherical displacements, strains and stresses
    # See Chapter 1.4.3 in Kausel's Elastodynamics

    # Symbol/Variable Definitions
    # (descriptions in parentheses refer to the symbols used by Kausel)
    #
    # Input:
    # - P[2]: Load vector (defined in cartesian coordinates x,y,z)
    # - G(3,3) (small): Green�s function for the frequency domain response in
    #                   direction i (defined in spherical coordinates R,f,t)
    #                   due to a unit load in direction j (defined in cartesian
    #                   coordinates x,y,z) 
    # - dGdx(3,3,3): derivative with respect to x[k] (defined in spherical
    #                coordinates R,f,t)
    # - m (greek): Shear modulus
    # - n (greek): Poisson's ratio
    # - R: Source�receiver distance in 3-D space
    # - f (greek): Angle of depression in spherical coordinates
    #
    # Output:
    # - u[2]: Displacement vector (in spherical coordinates R,f,t)
    # - s[5] (greek): Stresses (see below for their order)
    # - e[5] (greek): Strains (see below for their order)
    #
    # Intermediate:
    # - l (greek): Lame constant
    # - dudx(3,3): Displacement gradients

    # Lame constant
    l = 2*m*n/(1-2*n)

    # Displacement Vector (3x1) - Using G[i,j]*P[j]
    u = np.zeros([3], dtype = 'complex_')
    for i in range(3):
        for j in range(3):
            u[i] = u[i]+G[i,j]*P[j]

    # Displacement Gradient (3x3) - Using dG[i,j]*P[j]/dx[k]
    dudx = np.zeros([3,3], dtype = 'complex_')
    for i in range(3):
        for k in range(3):
            for j in range(3):
                dudx[i,k] += dGdx[i,j,k]*P[j]

    # Strain Vector (6x1) - Eqns 1.85
    e = np.zeros(6, dtype = 'complex_')
    e[0] = dudx[0,0] #R
    e[1] = u[0]/R+dudx[1,1]/R #f (phi)
    e[2] = u[0]/R+(1/np.tan(f))*u[1]/R+(1/np.sin(f))*dudx[2,2]/R #t (theta)
    e[3] = (1/np.sin(f))*dudx[1,2]/R+dudx[2,1]/R-(1/np.tan(f))*u[2]/R #ft
    e[4] = (1/np.sin(f))*dudx[0,2]/R+dudx[2,0]/R-u[2]/R #Rt
    e[5] = dudx[0,1]/R+dudx[1,0]-u[1]/R #Rf

    # Stress Vector (6x1) - Eqns 1.86
    s = np.zeros(6, dtype = 'complex_')
    evol = e[0]+e[1]+e[2]
    s[0] = 2*m*e[0]+l*evol #R
    s[1] = 2*m*e[1]+l*evol #f (phi)
    s[2] = 2*m*e[2]+l*evol #t (theta)
    s[3] = m*e[3] #ft
    s[4] = m*e[4] #Rt
    s[5] = m*e[5] #Rf

    # Switch to Geotechnical Notation
    e = -e
    s = -s

    return [u,e,s]



def analysis_cartesian(P,G,dGdx,m,n):
    # Cartesian displacements, strains and stresses
    # See Chapter 1.4.1 in Kausel's Elastodynamics

    # Symbol/Variable Definitions
    # (descriptions in parentheses refer to the symbols used by Kausel)
    #
    # Input:
    # - P[2]: Load vector
    # - G(3,3) (small): Green�s function for the frequency domain response in
    #                   direction i due to a unit load in direction j
    # - dGdx(3,3,3): derivative with respect to x[k]
    # - m (greek): Shear modulus
    # - n (greek): Poisson's ratio
    #
    # Output:
    # - u[2]: Displacement vector (in spherical coordinates R,f,t)
    # - s[5] (greek): Stresses (see below for their order)
    # - e[5] (greek): Strains (see below for their order)
    #
    # Intermediate:
    # - l (greek): Lame constant
    # - dudx(3,3): Displacement gradients

    # Lame constant
    l = 2*m*n/(1-2*n)

    # Displacement Vector (3x1) - Using G[i,j]*P[j]
    u = np.zeros(3, dtype = 'complex_')
    for i in range(3):
        for j in range(3):
            u[i] = u[i]+G[i,j]*P[j]

    # Displacement Gradient (3x3) - Using dG[i,j]*P[j]/dx[k]
    dudx = np.zeros([3,3], dtype = 'complex_')
    for i in range(3):
        for k in range(3):
            for j in range(3):
                dudx[i,k] += dGdx[i,j,k]*P[j]

    # Strain Vector (6x1) - Eqns 1.46 & 1.48
    e = np.zeros(6, dtype = 'complex_')
    e[0] = dudx[0,0] #x
    e[1] = dudx[1,1] #y
    e[2] = dudx[2,2] #z
    e[3] = dudx[2,1]+dudx[1,2] #yz
    e[4] = dudx[0,2]+dudx[2,0] #zx
    e[5] = dudx[1,0]+dudx[0,1] #xy

    # Stress Vector (6x1) - Eqns 1.53 & 1.57
    s = np.zeros(6, dtype = 'complex_')
    evol = e[0]+e[1]+e[2]
    s[0] = 2*m*e[0]+l*evol #x
    s[1] = 2*m*e[1]+l*evol #y
    s[2] = 2*m*e[2]+l*evol #z
    s[3] = m*e[3] #yz
    s[4] = m*e[4] #zx
    s[5] = m*e[5] #xy

    # Switch to Geotechnical Notation
    e = -e
    s = -s

    return [u,e,s]