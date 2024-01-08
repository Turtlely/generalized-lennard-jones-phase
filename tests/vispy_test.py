import numpy as np
import vispy
import vispy.scene
from vispy.scene import visuals
from vispy import app
import sys

canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# generate data
def solver(t):
    pos = np.array([[0.5 + t/1000, 0.5, 0], [0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0]])
    return pos
# These are the data that need to be updated each frame --^

scatter = visuals.Markers()
view.add(scatter)


#view.camera = scene.TurntableCamera(up='z')

# just makes the axes
axis = visuals.XYZAxis(parent=view.scene)


t = 0.0
def update(ev):
    global scatter
    global t
    t += 1.0
    scatter.set_data(solver(t), edge_color=None, face_color=(1, 1, 1, .5), size=10)

timer = app.Timer()
timer.connect(update)
timer.start(0.05)
if __name__ == '__main__':
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()