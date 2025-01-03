from mayavi import mlab
import os
import sys
import numpy as np

GAMMA = 4.0

assert len(sys.argv) == 2, "Please pass single filename."
assert os.path.isfile(sys.argv[1])
assert sys.argv[1].endswith(".npy")

v = np.load(sys.argv[1])
v = np.nan_to_num(v)
v[v > 1.0] = 0.0
v = np.power(v, 1.0/GAMMA)

mlab.figure(1, bgcolor=(0, 0, 0), size=(750, 750))
mlab.clf()

source = mlab.pipeline.scalar_field(v)
_min, _max = v.min(), v.max()
lower_thresh, upper_thresh = 0.15, 0.9
vol = mlab.pipeline.volume(source, vmin=_min + lower_thresh * (_max - _min), vmax=_min + upper_thresh * (_max - _min))
#vol = mlab.pipeline.volume(source)
mlab.show()
