import roboverse
import numpy as np

#env = roboverse.make('Widow250DrawerRandomizedOpen-v0', gui=True)
env = roboverse.make('Widow250BallBalancing-v0', gui=True)
env.reset()
for _ in range(250):
    #env.step(np.asarray([0., 0., 0., 0., 0., 0., 0., 0.]))
    action = env.action_space.sample()
    action = np.concatenate((action[:-2],  np.asarray([0., 0.])))
    #print(action)
    env.step(action)