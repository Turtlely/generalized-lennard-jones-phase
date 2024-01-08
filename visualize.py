import vispy
import vispy.scene
from vispy.scene import visuals
from vispy import app
import sys
import numpy as np
import constants
from vispy.scene.cameras import TurntableCamera
import vispy.scene as scene
import argparse
import glob
import os
# Folder name argument
parser = argparse.ArgumentParser("Visualization Application")
parser.add_argument("Name", nargs='?',default=max(glob.glob('simulations/*'), key=os.path.getctime),help="The name of the folder containing the simulation files. This should be a time stamp of when the simulation was finished")
args = parser.parse_args()
timestamp = args.Name

if timestamp[:12] == 'simulations/':
    timestamp = timestamp[12:]

# Window Size
WINDOW_SIZE = constants.WINDOW_SIZE # 10,000 x 10,000 picometer box, 10 by 10 nanometers

# Animation
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

# Camera control
view.camera = TurntableCamera(up='z',elevation=45,azimuth=135,fov=60,distance=2*WINDOW_SIZE*np.sqrt(3))

# Create scatterplot
scatter = visuals.Markers()

# Add scatterplot
view.add(scatter)

# Add axis
xax = scene.Axis(pos=[[0, 0], [WINDOW_SIZE, 0]], tick_direction=(0, -1), axis_color='r', tick_color='r', text_color='r', font_size=16, parent=view.scene)
yax = scene.Axis(pos=[[0, 0], [0, WINDOW_SIZE]], tick_direction=(-1, 0), axis_color='g', tick_color='g', text_color='g', font_size=16, parent=view.scene)
zax = scene.Axis(pos=[[0, 0], [-WINDOW_SIZE, 0]], tick_direction=(0, -1), axis_color='b', tick_color='b', text_color='b', font_size=16, parent=view.scene)
zax.transform = scene.transforms.MatrixTransform()  # its acutally an inverted xaxis
zax.transform.rotate(90, (0, 1, 0))  # rotate cw around yaxis
zax.transform.rotate(-45, (0, 0, 1))  # tick direction towards (-1,-1)

print("Plot set up")

# Import data file
print("Loading File")
XY = np.load(f"simulations/{timestamp}/position_log.npy", allow_pickle=True)

print("Done Loading File!")


frames = int(constants.t_total/constants.dt)
t = 0
def update(ev):
    global scatter
    global t
    t += 1

    if t > frames-1:
        t=0

    scatter.set_data(XY[t], edge_color=None, face_color=(1, 1, 1, 0.5), size=15)

timer = app.Timer()
timer.connect(update)
timer.start(1/constants.FPS)
if __name__ == '__main__':
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()