
import subprocess
from constants import STARTING_PORT
from constants import TOTAL_NUM_PORTS


processes = []
p = subprocess.Popen(["python", "train_stable_baselines.py", "falcon"])
processes.append(p)
p = subprocess.Popen(["python", "train_stable_baselines.py", "falco"])
processes.append(p)
p = subprocess.Popen(["python", "train_stable_baselines.py", "fox"])
processes.append(p)
p = subprocess.Popen(["python", "train_stable_baselines.py", "marth"])
processes.append(p)

for proc in  processes:
    proc.communicate() #now wait plus that you can send commands to process