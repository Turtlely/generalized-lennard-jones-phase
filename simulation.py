''' 
Simulation Units 

mass: AMU
length: pm
time: ps
Temperature: K
'''
#import numpy as np
import numpy as np
import time
import pandas as pd
import constants
import potentials
import partical_init
import os
import datetime
import scipy.spatial as spatial

def time_step(i,X,Y,VX,VY,M,FX,FY,X_new,Y_new,N,temperature_log,energy_log,pos_log,vel_log):
    # Start timer
    start_time = time.time()
    KE = 0
    T = 0
    U = 0

    # Construct KDTree for nearest neighbor search
    tree = spatial.cKDTree(np.c_[X,Y])
    groups = tree.query_ball_point(np.c_[X,Y], constants.rc)
    
    for n in range(N):
        xi = X[n]
        yi = Y[n]
        vx = VX[n]
        vy = VY[n]
        m = M[n] 
        k = groups[n]
        
        # Solve for the force using the current position
        r_array = np.sqrt(np.square(X[k]-xi)+np.square(Y[k]-yi))
        
        U += np.sum(potentials.V(r_array,constants.A,constants.B))/2

        F_ij_array = potentials.F(r_array,constants.A,constants.B)
        F_x = np.sum(F_ij_array* np.divide((xi - X[k]), r_array, out=np.zeros_like(X[k]), where=r_array!=0))
        F_y = np.sum(F_ij_array* np.divide((yi - Y[k]), r_array, out=np.zeros_like(Y[k]), where=r_array!=0))

        FX[n] = F_x
        FY[n] = F_y
        
        # New position
        xi += vx*constants.dt + 0.5*F_x/m*constants.dt**2 
        yi += vy*constants.dt + 0.5*F_y/m*constants.dt**2

        X_new[n] = xi
        Y_new[n] = yi

        KE+=0.5*m*(vx**2+vy**2)
        T+=m*(vx**2+vy**2)/(constants.kb*(3*N-3))

        
    X = np.copy(X_new)
    Y = np.copy(Y_new)
    
    for n in range(N):
        xi = X[n]
        yi = Y[n]
        vx = VX[n]
        vy = VY[n]
        m = M[n] 
        k = np.array(groups[n])

        # Solve for the new force using the updated position
        r_array = np.sqrt(np.square(X[k]-xi)+np.square(Y[k]-yi))
        F_ij_array = potentials.F(r_array,constants.A,constants.B)
        F_x = np.sum(F_ij_array* np.divide((xi - X[k]), r_array, out=np.zeros_like(X[k]), where=r_array!=0))
        F_y = np.sum(F_ij_array* np.divide((yi - Y[k]), r_array, out=np.zeros_like(Y[k]), where=r_array!=0))
        
        # FX/Y[n] returns the force that the particle felt at the current position
        FX_old = FX[n]
        FY_old = FY[n]

        # Update velocity
        vx += 0.5*(F_x+FX_old)/m * constants.dt
        vy += 0.5*(F_y+FY_old)/m * constants.dt
        
        # Hard boundary
        if xi < -constants.WINDOW_SIZE/2 or xi > constants.WINDOW_SIZE/2:
            vx*=-1
        if yi < - constants.WINDOW_SIZE/2 or yi > constants.WINDOW_SIZE/2:
            vy*=-1
        
        VX[n] = vx
        VY[n] = vy

    # Log X and Y
    pos_log[i] = np.c_[X,Y]

    # Log VX and VY of each particle
    vel_log[i] = np.c_[VX,VY]
    
    # Log hamiltonian
    H = (KE + U)*1e-7
    energy_log[i] = H
    print("Hamiltonian (eV): ", round(H,5))

    # Log Temperature
    print("Temperature (K): ", round(T,2))
    temperature_log[i] = T

    # Elapsed time and frames
    print("Elapsed Time (ps): ", round(constants.dt*i,4))
    print("Elapsed Frames: ", i)

    #FPS
    print("FPS: ", 1/(time.time() - start_time)) 
    print()

    return X,Y,VX,VY,M

def start_simulation(X,Y,VX,VY,M):
    # Initial Conditions
    FX = np.empty(constants.Nw*constants.Nh)
    FY = np.empty(constants.Nw*constants.Nh)

    X_new = np.empty_like(X)
    Y_new = np.empty_like(Y)

    # Number of particles
    N=constants.Nw*constants.Nh

    # Logging variables
    pos_log = np.empty(int(constants.t_total/constants.dt),dtype=np.ndarray)
    energy_log = np.empty(int(constants.t_total/constants.dt),dtype=np.float64)
    temperature_log = np.empty(int(constants.t_total/constants.dt),dtype=np.float64)
    vel_log = np.empty(int(constants.t_total/constants.dt),dtype=np.ndarray)
    
    # Simulation loop
    for i in range(int(constants.t_total/constants.dt)):
        X,Y,VX,VY,M = time_step(i,X,Y,VX,VY,M,FX,FY,X_new,Y_new,N,temperature_log,energy_log,pos_log,vel_log)

    # Save data
    print("DONE WITH SIMULATION, SAVING")
    dirPath = f"{os.getcwd()}/simulations/{datetime.datetime.now()}"
    os.mkdir(dirPath)

    np.save(f"{dirPath}/position_log",pos_log)
    np.save(f"{dirPath}/energy_log",energy_log)
    np.save(f"{dirPath}/temperature_log",temperature_log)
    np.save(f"{dirPath}/velocity_log",vel_log)
