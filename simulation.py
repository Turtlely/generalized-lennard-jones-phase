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

# Periodic Boundaries
x_ = np.linspace(-1, 1, 3)
x,y,z = np.meshgrid(x_, x_, x_,indexing='ij')
pbx = np.insert(np.delete(x.ravel(),13),0,0)*constants.WINDOW_SIZE
pby = np.insert(np.delete(y.ravel(),13),0,0)*constants.WINDOW_SIZE
pbz = np.insert(np.delete(z.ravel(),13),0,0)*constants.WINDOW_SIZE

# Time step
def time_step(i,X,Y,Z,X_new,Y_new,Z_new,VX,VY,VZ,M,T_desired):
    # Start timer
    start_time = time.time()
    KE = 0
    T = 0
    U = 0
    P = 0

    # Construct KDTree for nearest neighbor search
    for l in range(1,27):
        x, y, z = (pbx[l],pby[l],pbz[l])
        X[(l)*N:(l+1)*N] = X[0:N]+x
        Y[(l)*N:(l+1)*N] = Y[0:N]+y
        Z[(l)*N:(l+1)*N] = Z[0:N]+z

    points = np.c_[X,Y,Z]
    tree = spatial.cKDTree(points)
    groups = tree.query_ball_point(points, constants.rc,workers=-1)
    
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

        # Periodic boundary conditions
        if xi < -constants.WINDOW_SIZE/2:
            xi+=constants.WINDOW_SIZE
        
        if xi > constants.WINDOW_SIZE/2:
            xi-=constants.WINDOW_SIZE

        if yi < -constants.WINDOW_SIZE/2:
            yi+=constants.WINDOW_SIZE
        
        if yi > constants.WINDOW_SIZE/2:
            yi-=constants.WINDOW_SIZE

        if zi < -constants.WINDOW_SIZE/2:
            zi+=constants.WINDOW_SIZE
        
        if zi > constants.WINDOW_SIZE/2:
            zi-=constants.WINDOW_SIZE

        X_new[n] = xi
        Y_new[n] = yi
        Z_new[n] = zi

        P += (xi*F_x + yi*F_y + zi*F_z)    
        #P += np.dot([np.array([xi,yi,zi]),np.array([F_x,F_y,F_z])])
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
        if constants.USE_THERMOSTAT and np.random.random() < constants.freq:
            # Sample velocity from a boltzman distribution
            stdev = np.sqrt(constants.kb*T_desired/m)
            vx = np.random.normal(loc = 0, scale = stdev)
            vy = np.random.normal(loc = 0, scale = stdev)
            vz = np.random.normal(loc = 0, scale = stdev)
        else:
            # Update velocity
            vx += 0.5*(F_x+FX_old)/m * constants.dt
            vy += 0.5*(F_y+FY_old)/m * constants.dt
            vz += 0.5*(F_z+FZ_old)/m * constants.dt

        VX[n] = vx
        VY[n] = vy
        VZ[n] = vz

    # Log X and Y and Z
    pos_log[i] = np.c_[X[0:N],Y[0:N],Z[0:N]]

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
    P_real = P/(3*(constants.WINDOW_SIZE)**3) + N*constants.kb*T*(constants.WINDOW_SIZE)**-3
    print("Pressure (Pa): ", P_real * 1.66054e-15)
    p_log[i] = P_real

    # Elapsed time and frames
    print("Elapsed Time (ps): ", round(constants.dt*i,4))
    print("Elapsed Frames: ", i)

    #FPS
    print("FPS: ", 1/(time.time() - start_time)) 
    print()

    return X,Y,Z,VX,VY,VZ,M

def start_simulation(X,Y,Z,VX,VY,VZ,M,T_t):
    X_new = np.empty_like(X)
    Y_new = np.empty_like(Y)
    Z_new = np.empty_like(Z)
    frames = int(constants.t_total/constants.dt)

    # Temperature profile function
    T_profile = T_t(np.arange(frames))

    # Simulation loop
    for i in range(frames):
        X,Y,Z,VX,VY,VZ,M = time_step(i,X,Y,Z,X_new,Y_new,Z_new,VX,VY,VZ,M,T_profile[i])

    # Save data
    print("DONE WITH SIMULATION, SAVING")
    dirPath = f"{os.getcwd()}/simulations/{datetime.datetime.now()}"
    os.mkdir(dirPath)

    np.save(f"{dirPath}/position_log",pos_log)
    np.save(f"{dirPath}/energy_log",energy_log)
    np.save(f"{dirPath}/temperature_log",temperature_log)
    np.save(f"{dirPath}/velocity_log",vel_log)
