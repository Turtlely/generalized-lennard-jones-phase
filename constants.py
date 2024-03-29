import numpy as np

# Time Step
dt = 0.002 #0.002 picosecond time step, 2 femtosecond time step

# Total simulation timeframe (ps)
t_total = 1

# FPS for rendering
FPS = 1000

# Window Size
WINDOW_SIZE = 2200 # 10,000 x 10,000 picometer box, 10 by 10 nanometers

# Initial grid parameters, aim for more than σ separation between molecules
Ni = 6
Nj = 6
Nk = 6

'''
Modelling an argon-like substance, Values taken from wikipedia
'''

# Boltzmann Constant (1.38E-23 J/K) kgm^2/s^2/K
kb = 8310 #AMU * pm^2 / ps^2 / K
σ = 340 # pm
ε = 120*kb # AMU * pm^2 / ps^2

# Lennard Jones Parameter
A = 4 * ε * σ**12
B = 4 * ε * σ**6

# Cutoff distance
rc = 2.5*σ

# Initial temperature (K)
T_init = 300

# Temperature profile as a function of time
def T_t(i):
    time = i*dt
    return T_init * np.exp(-1*time) + 0#1600*time*(1-time)

# Coupling constant to a heat bath, cc = freq/dt
# 1 collision every 1000 time steps
freq = 1/100

USE_THERMOSTAT = False #True