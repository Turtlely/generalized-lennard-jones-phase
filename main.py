# Imports
import simulation as sim
import partical_init as pi
import constants

# Initialize
X,Y,Z,VX,VY,VZ,M = pi.createParticlesCube(constants.Ni,constants.Nj,constants.Nk,1800,1800,1800)

# Start Simulation
sim.start_simulation(X,Y,Z,VX,VY,VZ,M,constants.T_t)