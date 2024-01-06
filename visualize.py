import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  
import pandas as pd
import numpy as np
import constants

# Window Size
WINDOW_SIZE = constants.WINDOW_SIZE # 10,000 x 10,000 picometer box, 10 by 10 nanometers

# Animation
fig = plt.figure()
ax = plt.axes(xlim=(-WINDOW_SIZE/2,WINDOW_SIZE/2),ylim=(-WINDOW_SIZE/2,WINDOW_SIZE/2))   
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel("Picometers")
ax.set_ylabel("Picometers")
plt.grid()

print("Loading File")
XY = np.load("simulations/2024-01-06 15:17:47.275782/position_log.npy", allow_pickle=True)
#XY2 = np.load("simulations/long simulation/position_data.npy", allow_pickle=True)

print("Done!")

frames = int(constants.t_total/constants.dt)

scat = ax.scatter(*zip(*XY[0]))

def animate(i):
    print(i)
    # Plot particles
    scat.set_offsets(XY[i])

print("Starting render")
anim = FuncAnimation(fig, animate, 
                     frames = frames, interval = 0.01) 
print("Finished render!")
plt.show()
   
#print("Saving")
#anim.save('Output.mp4', writer = 'ffmpeg', fps = 500,dpi=300) 
#print("Finished saving!")