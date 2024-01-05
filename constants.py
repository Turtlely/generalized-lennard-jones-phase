# Time Step
dt = 0.002 #0.002 picosecond time step, 2 femtosecond time step

# Total simulation timeframe (ps)
t_total = 10

# Window Size
WINDOW_SIZE = 10000 # 10,000 x 10,000 picometer box, 10 by 10 nanometers

# Initial grid parameters, aim for more than σ separation between molecules
Nw = 4
Nh = 4

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