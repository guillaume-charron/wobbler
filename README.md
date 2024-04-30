# Wobbler: Ball balancing robot
A ball balancing environment utilizing [pybullet](https://github.com/bulletphysics/bullet3) for simulation of robotic manipulation tasks. 

## Setup
I recommend using [conda](https://docs.anaconda.com/anaconda/install/) for setup:

```
conda create -n roboverse python=3.9
conda activate roboverse
pip install -r requirements.txt
```
When using this repository with other projects, run `pip install -e .` in the root directory of this repo. 

## Reproduce experiments

To reproduce all the data stored in the `runs` folder, simply execute the `run_commands.py` script.

## Run on real hardware

To run the policy on real hardware, simply change the environment in the config file for `Widow250EnvROSARob-v0`, specify the pretrained model in the `from_pretrained` field and set `real_application` field to True.

## Credits
Roboverse developers: [Avi Singh](https://www.avisingh.org/), Albert Yu, Jonathan Yang, [Michael Janner](https://people.eecs.berkeley.edu/~janner/), Huihan Liu, Gaoyue Zhou
