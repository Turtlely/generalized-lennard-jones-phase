import numpy as np
import constants

def createParticlesCube(a,b,c,i,j,k):
    X = np.empty(a*b*c)
    Y = np.empty(a*b*c)
    Z = np.empty(a*b*c)
    VX = np.empty(a*b*c)
    VY = np.empty(a*b*c)
    VZ = np.empty(a*b*c)
    M = np.ones(a*b*c)
    
    N = a*b*c

    # Assign random velocities
    VX = np.random.random(N)-0.5
    VY = np.random.random(N)-0.5
    VZ = np.random.random(N)-0.5

    # Find center of mass velocity
    vx_cm = np.average(VX)
    vy_cm = np.average(VY)
    vz_cm = np.average(VZ)

    # Adjust initial velocities to have zero momentum
    VX -= vx_cm
    VY -= vy_cm
    VZ -= vz_cm

    # Current temperature
    T_i = np.sum(M*(np.power(VX,2)+np.power(VY,2)+np.power(VZ,2))/(constants.kb*(3*N)))

    # Scale velocities
    factor = (constants.T_init/T_i)**0.5
    VX *= factor
    VY *= factor
    VZ *= factor

    n=0
    for x in np.linspace(-i/2,i/2,a):
        for y in np.linspace(-j/2,j/2,b):
            for z in np.linspace(-k/2,k/2,c):
                X[n] = x
                Y[n] = y
                Z[n] = z
                n+=1
    # Return
    return X,Y,Z,VX,VY,VZ,M

    