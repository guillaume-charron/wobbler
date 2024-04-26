import subprocess
import concurrent.futures

commands = [
    "python run.py meta.add_to_runname=\"8envs_10Msteps_norand_nogcrl\" environment.gcrl=False meta.randomize_after_steps=11000000",
    "python run.py meta.add_to_runname=\"8envs_10Msteps_onlyrand_nogcrl\" environment.gcrl=False meta.randomize_after_steps=0",
    "python run.py meta.add_to_runname=\"8envs_10Msteps_7_5Mrand_nogcrl\" environment.gcrl=False",
    "python run.py meta.add_to_runname=\"8envs_10Msteps_7_5Mrand_gcrl\" environment.gcrl=True",
    "python run.py meta.add_to_runname=\"8envs_10Msteps_onlyrand_gcrl\" environment.gcrl=True meta.randomize_after_steps=0",
    "python run.py meta.add_to_runname=\"8envs_10Msteps_norand_gcrl\" environment.gcrl=True meta.randomize_after_steps=11000000"
]

def run_command(command):
    print(f"Running command: {command}")
    terminal_command = f'gnome-terminal -- bash -c "{command}; exec bash"'
    subprocess.run(terminal_command, shell=True)
    print("Command finished")
    
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    executor.map(run_command, commands)