#Particle attributes
# Mass
# Velocity

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  

# Time Step
dt = 0.1/3

dist = 3

drift_log = []
pos = []
accel = []

def V(r,A=1,B=1):
    return A/r**12 - B/r**6

def F(r,A=1,B=1):
    return 12*A*r**-13 - 6*B*r**-7

class Particle:
    def __init__(self, x, y, m):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.m = m
    def get_KE(self):
        return 0.5*self.m*np.sqrt(self.vx**2+self.vy**2)

# Particle list
p_list = [Particle(-dist/2,0,1),Particle(dist/2,0,1)]

# Returns list of particles to interact with.
# Naive algorithm: Consider interactions with all particles
def pairwise(p):
    return [p_j for p_j in p_list if p_j != p]

def animate(i):
    ax.clear()

    x = []
    y = []

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
            r = np.sqrt((p_i.x-p_j.x)**2 - (p_i.y-p_j.y)**2)

            # Energy bookkeeping calculation:
            U += V(r)/2 # (divide by two to account for double counting)

            # Compute force acting on p_i, due to p_j
            F_ij = F(r)

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
    
    # Energy bookkeeping calculations:
    #print("Kinetic energy: ", KE)
    #print("Potential energy: ", U)
    #print("Hamiltonian: ", KE + U)
    #print("Initial Energy: ", V(1.5))
    print("Energy Drift: ", (KE+U) - V(dist))
    
    drift_log.append((KE+U) - V(dist))
    pos.append(p_list[0].x)

    # Plot (Maybe move this out, so plotting doesnt slow down the simulation?)
    ax.scatter(x,y)
    plt.grid()
    plt.xlim([-5,5])
    plt.ylim([-5,5])
    ax.set_aspect('equal', adjustable='box')

fig = plt.figure()
ax = plt.axes(xlim=(-2,2),ylim=(-2,2))   

ani = FuncAnimation(fig,animate,interval=1)
plt.show()

fig2 = plt.figure()
plt.plot(range(len(drift_log)),drift_log)
plt.plot(range(len(pos)),pos)

plt.show()
#print(drift_log)





'''
Research:
To minimize energy drift, you can either:
1. lower the time step
2. bring the particles closer to the equilibrium point
3. increase the mass, which decreases the total acceleration

It seems the simulation is stable when the maximum acceleration is below some threshold. Above this, the particle gains energy too fast
'''