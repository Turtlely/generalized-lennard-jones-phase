import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  
import pandas as pd
import numpy as np
import constants

# Window Size
WINDOW_SIZE = 10000 # 10,000 x 10,000 picometer box, 10 by 10 nanometers

# Animation
fig = plt.figure()
ax = plt.axes(xlim=(-WINDOW_SIZE/2,WINDOW_SIZE/2),ylim=(-WINDOW_SIZE/2,WINDOW_SIZE/2))   
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel("Picometers")
ax.set_ylabel("Picometers")
plt.grid()

XY = np.load("position_data.npy", allow_pickle=True)

frames = int(constants.t_total/constants.dt)

scat = ax.scatter(*zip(*XY[0]))

def animate(i):
    # Plot particles
    scat.set_offsets(XY[i])

anim = FuncAnimation(fig, animate, 
                     frames = frames, interval = 1) 
  
   
anim.save('Output.mp4',  
          writer = 'ffmpeg', fps = 500,dpi=300) 