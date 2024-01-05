#Particle attributes
# Mass
# Velocity

''' 
Simulation Units 

mass: AMU
length: pm
time: ps
Temperature: K
'''

import numpy as np
import time
import pandas as pd
import constants
import potentials
import partical_init

# Time Step
dt = constants.dt

# Total simulation timeframe (ps)
t_total = constants.t_total

# Window Size
WINDOW_SIZE = constants.WINDOW_SIZE 

# Initial grid parameters, aim for more than σ separation between molecules
Nw = constants.Nw
Nh = constants.Nh

# Number of particles
N=Nw*Nh

'''
Modelling an argon-like substance, Values taken from wikipedia
'''

# Boltzmann Constant
kb = constants.kb
σ = constants.σ
ε = constants.ε

# Lennard Jones Parameter
A = constants.A
B = constants.B

# Creating the vectors
X,Y,VX,VY,FX,FY,M = partical_init.createParticlesGrid(Nw,Nh,WINDOW_SIZE/2,WINDOW_SIZE/2)

X_new = np.empty_like(X)
Y_new = np.empty_like(Y)

# Logging variables
pos_log = np.empty(int(t_total/dt),dtype=np.ndarray)

# Simulation Frame
def time_step(i):
    # Start timer
    start_time = time.time()
    global X, Y, VX, VY
    
    KE = 0
    T = 0

    for n in range(N):
        xi = X[n]
        yi = Y[n]
        vx = VX[n]
        vy = VY[n]
        m = 1 

        # Kinetic Energy Calculation
        KE += 0.5*m*(vx**2+vy**2)
        # Temperature calculation
        T += m*(vx**2+vy**2)/(kb*(3*N-3))
        
        # Solve for the force using the current position

        r_array = np.sqrt(np.square(X-xi)+np.square(Y-yi))
        U = np.sum(potentials.V(r_array,A,B))/2
        F_ij_array = potentials.F(r_array,A,B)
        F_x = np.sum(F_ij_array* np.divide((xi - X), r_array, out=np.zeros_like(X), where=r_array!=0))
        F_y = np.sum(F_ij_array* np.divide((yi - Y), r_array, out=np.zeros_like(Y), where=r_array!=0))
        
        FX[n] = F_x
        FY[n] = F_y

        # New position
        xi += vx*dt + 0.5*F_x/m*dt**2 
        yi += vy*dt + 0.5*F_y/m*dt**2
        
        # Periodic/ish boundary
        '''
        if xi < -WINDOW_SIZE/2:
            xi += WINDOW_SIZE
        if xi > WINDOW_SIZE/2:
            xi -= WINDOW_SIZE
        if yi < - WINDOW_SIZE/2:
            yi += WINDOW_SIZE
        if yi > WINDOW_SIZE/2:           
            yi -= WINDOW_SIZE
        '''

        X_new[n] = xi
        Y_new[n] = yi

    X = np.copy(X_new)
    Y = np.copy(Y_new)

    for n in range(N):
        xi = X[n]
        yi = Y[n]
        vx = VX[n]
        vy = VY[n]
        m = 1 

        # Solve for the new force using the updated position

        r_array = np.sqrt(np.square(X-xi)+np.square(Y-yi))
        U = np.sum(potentials.V(r_array,A,B))/2
        F_ij_array = potentials.F(r_array,A,B)
        F_x = np.sum(F_ij_array* np.divide((xi - X), r_array, out=np.zeros_like(X), where=r_array!=0))
        F_y = np.sum(F_ij_array* np.divide((yi - Y), r_array, out=np.zeros_like(Y), where=r_array!=0))
        
        # FX/Y[n] returns the force that the particle felt at the current position
        FX_old = FX[n]
        FY_old = FY[n]

        vx += 0.5*(F_x+FX_old)/m * dt
        vy += 0.5*(F_y+FY_old)/m * dt
        
        # Hard boundary
        
        if xi < -WINDOW_SIZE/2 or xi > WINDOW_SIZE/2:
            vx*=-1
        if yi < - WINDOW_SIZE/2 or yi > WINDOW_SIZE/2:
            vy*=-1
        
        VX[n] = vx
        VY[n] = vy
    
    # Log X and Y
    XY = np.c_[X,Y]
    pos_log[i] = XY

    # Log VX and VY of each particle
    #VX
    #VY

    # Log hamiltonian, temperature, pressure
    H = (KE + U)*1e-7
    #T = T

    # Log FPS
    fps = 1/(time.time() - start_time)

    print("FPS: ", fps)
    print("Hamiltonian (eV): ", round(H,5))
    print("Elapsed Time (ps): ", round(dt*i,4))
    print("Frames Elapsed: ", i)

# Simulation loop

for i in range(int(t_total/dt)):
    time_step(i)

np.save("position_data",pos_log)
#print("Position Log")
#print(pos_log[1][1])