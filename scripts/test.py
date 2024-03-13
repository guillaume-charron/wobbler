import roboverse
import numpy as np

#env = roboverse.make('Widow250DrawerRandomizedOpen-v0', gui=True)
env = roboverse.make('Widow250BallBalancing-v0', gui=True)
env.reset()
for _ in range(50):
    env.step(np.asarray([0, 0., 0., 0., 0., 0., 0., 0.]))