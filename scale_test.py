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

# Performance tracking parameters
interval = 1

# Time Step
dt = 0.01 #0.01 picosecond time step, 10 femtosecond time step

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


# Particle class
class Particle:
    def __init__(self, x, y, m):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.m = m
    def get_KE(self):
        return 0.5*self.m*np.sqrt(self.vx**2+self.vy**2)

# Number of particles
N=100

# Particle List
p_list = [Particle(WINDOW_SIZE*np.random.random()-WINDOW_SIZE/2,WINDOW_SIZE*np.random.random()-WINDOW_SIZE/2,1) for n in range(N)] # Currently randomly sticks particles too close together causing them to fly apart, so high densities does not work

# Returns list of particles to interact with.
# Naive algorithm: Consider interactions with all particles
# TODO could optimize this function
def pairwise(p):
    return [p_j for p_j in p_list if p_j != p]

# Simulation Frame
def animate(i):
    # Start timer
    if i%interval == 0:
        start_time = time.time()

    # Clear previous frame
    ax.clear()

    # List to store particle coordinates
    x = []
    y = []

    # Energy tracking variables
    KE = 0
    U = 0

    # Calculate updated velocities for all particles
    for p_i in p_list:
        # Energy bookkeeping calculations:
        KE += p_i.get_KE()

        # Net force components
        F_x = 0
        F_y = 0

        for p_j in pairwise(p_i):
            # Compute distance between p_i and p_j
            r = np.sqrt((p_i.x-p_j.x)**2 + (p_i.y-p_j.y)**2)

            # Energy bookkeeping calculation:
            U += V(r,A,B)/2 # (divide by two to account for double counting)

            # Compute force acting on p_i, due to p_j
            F_ij = F(r,A,B)

            # Components of force
            F_ij_x = F_ij * (p_i.x - p_j.x)/r
            F_ij_y = F_ij * (p_i.y - p_j.y)/r

            F_x += F_ij_x
            F_y += F_ij_y

        # Update velocities
        p_i.vx += F_x/p_i.m * dt
        p_i.vy += F_y/p_i.m * dt

    # Update positions for all particles using new updated velocities
    for p in p_list:
        # Update
        p.x += p.vx * dt
        p.y += p.vy * dt
        
        x.append(p.x)
        y.append(p.y)

    # Plot particles
    ax.scatter(x,y)
    plt.grid()
    plt.xlim([-WINDOW_SIZE/2,WINDOW_SIZE/2])
    plt.ylim([-WINDOW_SIZE/2,WINDOW_SIZE/2])
    ax.set_aspect('equal', adjustable='box')

    # FPS counter
    if i%interval == 0:
        print("FPS: ", interval/(time.time() - start_time))
    
    # Energy bookkeeping calculations:
    #print("Kinetic energy: ", KE)
    #print("Potential energy: ", U)
    #print("Hamiltonian: ", KE + U)
    #print("Initial Energy: ", V(1.5))
    #print("Energy Drift: ", (KE+U) - V(dist))

# Animation
fig = plt.figure()
ax = plt.axes(xlim=(-WINDOW_SIZE/2,WINDOW_SIZE/2),ylim=(-WINDOW_SIZE/2,WINDOW_SIZE/2))   

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