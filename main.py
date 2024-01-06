# Imports
import simulation as sim
import partical_init as pi
import constants

# Initialize
X,Y,VX,VY,M = pi.createParticlesGrid(constants.Nw,constants.Nh,constants.WINDOW_SIZE/2,constants.WINDOW_SIZE/2)
#sim = sim.simulation(X,Y,VX,VY,M)

# Start Simulation
sim.start_simulation(X,Y,VX,VY,M)