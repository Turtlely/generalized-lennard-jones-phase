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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  
import time

# Time Step
dt = 0.002 #0.002 picosecond time step, 2 femtosecond time step

# Window Size
WINDOW_SIZE = 10000 # 10,000 x 10,000 picometer box, 10 by 10 nanometers

'''
Modelling an argon-like substance, Values taken from wikipedia
'''

# Boltzmann Constant (1.38E-23 J/K) kgm^2/s^2/K
kb = 8310 #AMU * pm^2 / ps^2 / K
σ = 340 # pm
ε = 120*kb # AMU * pm^2 / ps^2

# Lennard Jones Parameters
A = 4 * ε * σ**12
B = 4 * ε * σ**6

# Lennard Jones function
def V(r,A=1,B=1):
    return A/r**12 - B/r**6

def F(r,A=1,B=1):
    return 12*A*r**-13 - 6*B*r**-7

# Return list of particles to compute forces with
def pair_selector(XY,n):
    return [XY[i] for i in range(len(XY)) if i != n]

# Particle generation function
# Return two vectors of N length containing X and Y position of each particle
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
        VX[n] = 500*np.random.random()-250
        VY[n] = 500*np.random.random()-250
        M[n] = 39.948
    return X,Y,VX,VY,FX,FY,M

def createParticlesGrid(a,b,c,d):
    X = np.empty(a*b)
    Y = np.empty(a*b)
    VX = np.empty(a*b)
    VY = np.empty(a*b)
    FX = np.empty(a*b)
    FY = np.empty(a*b)
    M = np.empty(a*b)
    
    n=0
    for i in np.linspace(-c/2,c/2,a):
        for j in np.linspace(-d/2,d/2,b):
            X[n] = i
            Y[n] = j
            VX[n] = 700*np.random.random()-350
            VY[n] = 700*np.random.random()-350
            M[n] = 39.948
            n+=1
    return X,Y,VX,VY,FX,FY,M

# Creating the vectors
#X,Y,VX,VY,FX,FY,M = createParticles(N)
X,Y,VX,VY,FX,FY,M = createParticlesGrid(7,7,2500,2500)

# Number of particles
N=49

# Animation
fig = plt.figure()
ax = plt.axes(xlim=(-WINDOW_SIZE/2,WINDOW_SIZE/2),ylim=(-WINDOW_SIZE/2,WINDOW_SIZE/2))   
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel("Picometers")
ax.set_ylabel("Picometers")
plt.grid()
scat = ax.scatter(X,Y)
XY = np.c_[X,Y]
XY_new = np.empty_like(XY)

# Simulation Frame
def animate(i):
    # Start timer
    start_time = time.time()
    global XY
    global VX
    global VY
    global M
    
    KE = 0
    U = 0
    T = 0

    for n in range(N):
        xi,yi = XY[n]
        vx = VX[n]
        vy = VY[n]
        m = M[n]

        # Kinetic Energy Calculation
        KE += 0.5*m*(vx**2+vy**2)
        T+= m*(vx**2+vy**2)/(kb*(3*N-3))

        
        # Solve for the force using the current position

        F_x = 0
        F_y = 0

        for xj,yj in pair_selector(XY,n):
            # Distance between the particles
            r = np.sqrt((xi-xj)**2+(yi-yj)**2)

            # Energy bookkeeping calculation:
            U += V(r,A,B)/2 # (divide by two to account for double counting)

            # Compute force acting on p_i, due to p_j
            F_ij = F(r,A,B)

            # Components of force
            F_ij_x = F_ij * (xi - xj)/r
            F_ij_y = F_ij * (yi - yj)/r

            F_x += F_ij_x
            F_y += F_ij_y
        
        FX[n] = F_x
        FY[n] = F_y
        
        # New position
        xi += vx*dt + 0.5*F_x/m*dt**2 
        yi += vy*dt + 0.5*F_y/m*dt**2

        XY_new[n] = (xi,yi)

    XY = np.copy(XY_new)

    for n in range(N):
        xi,yi = XY[n]
        vx = VX[n]
        vy = VY[n]
        m = M[n]

        # Solve for the new force using the updated position

        F_x = 0
        F_y = 0

        for xj,yj in pair_selector(XY,n):
            # Distance between the particles
            r = np.sqrt((xi-xj)**2+(yi-yj)**2)
            # Compute force acting on p_i, due to p_j
            F_ij = F(r,A,B)

            # Components of force
            F_ij_x = F_ij * (xi - xj)/r
            F_ij_y = F_ij * (yi - yj)/r

            F_x += F_ij_x
            F_y += F_ij_y
        
        # FX/Y[n] returns the force that the particle felt at the current position
        FX_old = FX[n]
        FY_old = FY[n]

        vx += 0.5*(F_x+FX_old)/m * dt
        vy += 0.5*(F_y+FY_old)/m * dt

        VX[n] = vx
        VY[n] = vy
    
    # Plot particles
    scat.set_offsets(XY)

    # FPS counter
    print("FPS: ", 1/(time.time() - start_time))

    # Energy Drift Analysis
    #print("Potential energy (eV): ", U*1e-7)
    #print("Kinetic energy (eV): ", KE*1e-7)
    print("Hamiltonian (eV): ", round((KE + U)*1e-7,2))
    print("Temperature (K): ", round(T,2))
    print("Elapsed Time (ps): ", round(dt*i,4))
    print()

ani = FuncAnimation(fig,animate,interval=1,cache_frame_data=False)
plt.show()




'''
Research:
To minimize energy drift, you can either:
1. lower the time step
2. bring the particles closer to the equilibrium point
3. increase the mass, which decreases the total acceleration

It seems the simulation is stable when the maximum acceleration is below some threshold. Above this, the particle gains energy too fast
'''