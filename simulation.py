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
import constants
import potentials
import partical_init
import os
import datetime
import scipy.spatial as spatial

# Initial Conditions
FX = np.empty(constants.Ni*constants.Nj*constants.Nk)
FY = np.empty(constants.Ni*constants.Nj*constants.Nk)
FZ = np.empty(constants.Ni*constants.Nj*constants.Nk)

# Number of particles
N=constants.Ni*constants.Nj*constants.Nk

# Logging variables
pos_log = np.empty(int(constants.t_total/constants.dt),dtype=np.ndarray)
vel_log = np.empty(int(constants.t_total/constants.dt),dtype=np.ndarray)
energy_log = np.empty(int(constants.t_total/constants.dt),dtype=np.float64)
temperature_log = np.empty(int(constants.t_total/constants.dt),dtype=np.float64)
p_log = np.empty(int(constants.t_total/constants.dt),dtype=np.float64)

def time_step(i,X,Y,Z,X_new,Y_new,Z_new,VX,VY,VZ,M):
    # Start timer
    start_time = time.time()
    KE = 0
    T = 0
    U = 0
    P = 0

    # Construct KDTree for nearest neighbor search
    tree = spatial.cKDTree(np.c_[X,Y,Z])
    groups = tree.query_ball_point(np.c_[X,Y,Z], constants.rc,workers=-1)
    
    for n in range(N):
        xi = X[n]
        yi = Y[n]
        zi = Z[n]
        vx = VX[n]
        vy = VY[n]
        vz = VZ[n]
        m = M[n] 
        k = groups[n]
        
        # Solve for the force using the current position
        r_array = np.sqrt(np.square(X[k]-xi)+np.square(Y[k]-yi)+np.square(Z[k]-zi))
        r_array_XY = np.sqrt(np.square(X[k]-xi)+np.square(Y[k]-yi))
        U += np.sum(potentials.V(r_array,constants.A,constants.B))

        F_ij_array = potentials.F(r_array,constants.A,constants.B)
        F_x = np.sum(F_ij_array* np.divide((xi - X[k]), r_array_XY, out=np.zeros_like(X[k]), where=r_array_XY!=0))
        F_y = np.sum(F_ij_array* np.divide((yi - Y[k]), r_array_XY, out=np.zeros_like(Y[k]), where=r_array_XY!=0))
        F_z = np.sum(F_ij_array* np.divide((zi - Z[k]), r_array, out=np.zeros_like(Z[k]), where=r_array!=0))

        FX[n] = F_x
        FY[n] = F_y
        FZ[n] = F_z
        
        # New position
        xi += vx*constants.dt + 0.5*F_x/m*constants.dt**2 
        yi += vy*constants.dt + 0.5*F_y/m*constants.dt**2
        zi += vz*constants.dt + 0.5*F_z/m*constants.dt**2

        X_new[n] = xi
        Y_new[n] = yi
        Z_new[n] = zi

        P += np.dot(F_ij_array,r_array)
        KE+=0.5*m*(vx**2+vy**2+vz**2)
        T+=m*(vx**2+vy**2+vz**2)/(constants.kb*(3*N))

        
    X = np.copy(X_new)
    Y = np.copy(Y_new)
    Z = np.copy(Z_new)

    for n in range(N):
        xi = X[n]
        yi = Y[n]
        zi = Z[n]
        vx = VX[n]
        vy = VY[n]
        vz = VZ[n]
        m = M[n] 
        k = np.array(groups[n])

        # Solve for the new force using the updated position
        r_array = np.sqrt(np.square(X[k]-xi)+np.square(Y[k]-yi)+np.square(Z[k]-zi))
        r_array_XY = np.sqrt(np.square(X[k]-xi)+np.square(Y[k]-yi))

        F_ij_array = potentials.F(r_array,constants.A,constants.B)
        F_x = np.sum(F_ij_array* np.divide((xi - X[k]), r_array_XY, out=np.zeros_like(X[k]), where=r_array_XY!=0))
        F_y = np.sum(F_ij_array* np.divide((yi - Y[k]), r_array_XY, out=np.zeros_like(Y[k]), where=r_array_XY!=0))
        F_z = np.sum(F_ij_array* np.divide((zi - Z[k]), r_array, out=np.zeros_like(Z[k]), where=r_array!=0))

        # FX/Y[n] returns the force that the particle felt at the current position
        FX_old = FX[n]
        FY_old = FY[n]
        FZ_old = FZ[n]

        # Anderson Thermostat

        # Probability of particle undergoing collision
        if np.random.random() < constants.freq:
            # Sample velocity from a boltzman distribution
            pass

        # Update velocity
        vx += 0.5*(F_x+FX_old)/m * constants.dt
        vy += 0.5*(F_y+FY_old)/m * constants.dt
        vz += 0.5*(F_z+FZ_old)/m * constants.dt
        
        # Hard boundary
        if xi < -constants.WINDOW_SIZE/2 or xi > constants.WINDOW_SIZE/2:
            vx*=-1
        if yi < -constants.WINDOW_SIZE/2 or yi > constants.WINDOW_SIZE/2:
            vy*=-1
        if zi < -constants.WINDOW_SIZE/2 or zi > constants.WINDOW_SIZE/2:
            vz*=-1

        VX[n] = vx
        VY[n] = vy
        VZ[n] = vz

    # Log X and Y and Z
    pos_log[i] = np.c_[X,Y,Z]

    # Log VX and VY and VZ of each particle
    vel_log[i] = np.c_[VX,VY,VZ]
    
    # Log hamiltonian
    U /= 2
    H = (KE + U)*1e-7
    energy_log[i] = H
    print("Hamiltonian (eV): ", round(H,5))

    # Log Temperature
    print("Temperature (K): ", round(T,2))
    temperature_log[i] = T

    # Log Pressure
    P_real = P*constants.WINDOW_SIZE**-3/3 + N*constants.kb*T*constants.WINDOW_SIZE**-3
    print("Pressure (Pa): ", P_real * 1.66054e-15)
    p_log[i] = P_real

    # Elapsed time and frames
    print("Elapsed Time (ps): ", round(constants.dt*i,4))
    print("Elapsed Frames: ", i)

    #FPS
    print("FPS: ", 1/(time.time() - start_time)) 
    print()

    return X,Y,Z,VX,VY,VZ,M

def start_simulation(X,Y,Z,VX,VY,VZ,M):
    X_new = np.empty_like(X)
    Y_new = np.empty_like(Y)
    Z_new = np.empty_like(Z)

    # Simulation loop
    for i in range(int(constants.t_total/constants.dt)):
        X,Y,Z,VX,VY,VZ,M = time_step(i,X,Y,Z,X_new,Y_new,Z_new,VX,VY,VZ,M)

    # Save data
    print("DONE WITH SIMULATION, SAVING")
    dirPath = f"{os.getcwd()}/simulations/{datetime.datetime.now()}"
    os.mkdir(dirPath)

    np.save(f"{dirPath}/position_log",pos_log)
    np.save(f"{dirPath}/energy_log",energy_log)
    np.save(f"{dirPath}/temperature_log",temperature_log)
    np.save(f"{dirPath}/velocity_log",vel_log)
