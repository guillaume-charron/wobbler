import roboverse
import numpy as np

#env = roboverse.make('Widow250DrawerRandomizedOpen-v0', gui=True)
env = roboverse.make('Widow250BallBalancing-v0', gui=True)
env.reset()
for _ in range(250):
    env.step(np.asarray([0., 0., 0., 0., 0., 0., 0., 0.]))
    #env.step(env.action_space.sample())