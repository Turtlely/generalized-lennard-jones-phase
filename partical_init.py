import numpy as np
import matplotlib.pyplot as plt
import constants

# Particle generation function
def createParticles(N):
    X = np.empty(N)
    Y = np.empty(N)
    VX = np.empty(N)
    VY = np.empty(N)
    FX = np.empty(N)
    FY = np.empty(N)
    M = np.empty(N)

    for n in range(N):
        x = WINDOW_SIZE*np.random.random()-WINDOW_SIZE/2
        y = WINDOW_SIZE*np.random.random()-WINDOW_SIZE/2
        X[n] = x
        Y[n] = y
        VX[n] = 0
        VY[n] = 0
        M[n] = 39.948
    return X,Y,VX,VY,FX,FY,M

def createParticlesGrid(a,b,c,d):
    X = np.empty(a*b)
    Y = np.empty(a*b)
    VX = np.empty(a*b)
    VY = np.empty(a*b)
    M = np.ones(a*b)
    
    N = a*b

    # Assign random velocities
    VX = np.random.random(N)-0.5
    VY = np.random.random(N)-0.5
    
    # Find center of mass velocity
    vx_cm = np.average(VX)
    vy_cm = np.average(VY)
    # Adjust initial velocities to have zero momentum
    VX -= vx_cm
    VY -= vy_cm

    # Current temperature
    T_i = np.sum(M*(np.power(VX,2)+np.power(VY,2))/(constants.kb*(3*N)))

    # Scale velocities
    factor = (constants.T_init/T_i)**0.5
    VX *= factor
    VY *= factor

    n=0
    for i in np.linspace(-c/2,c/2,a):
        for j in np.linspace(-d/2,d/2,b):
            X[n] = i
            Y[n] = j
            n+=1
    return X,Y,VX,VY,M

    