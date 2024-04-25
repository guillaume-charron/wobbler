import subprocess

commands = [
    "python run.py meta.add_to_runname=\"64envs_25Msteps_norand_nogcrl\" environment.gcrl=False meta.randomize_after_steps=26000000",
    "python run.py meta.add_to_runname=\"64envs_25Msteps_onlyrand_nogcrl\" environment.gcrl=False meta.randomize_after_steps=0",
    "python run.py meta.add_to_runname=\"64envs_25Msteps_10Mrand_nogcrl\" environment.gcrl=False",
    "python run.py meta.add_to_runname=\"64envs_25Msteps_10Mrand_gcrl\" environment.gcrl=True",
    "python run.py meta.add_to_runname=\"64envs_25Msteps_onlyrand_gcrl\" environment.gcrl=True meta.randomize_after_steps=0",
    "python run.py meta.add_to_runname=\"64envs_25Msteps_norand_gcrl\" environment.gcrl=True meta.randomize_after_steps=26000000"
]

for command in commands:
    print(f"Running command: {command}")
    subprocess.run(command, shell=True)
    print("Command finished")