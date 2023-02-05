import subprocess
from subprocess import Popen
from constants import STARTING_PORT


TOTAL_NUM_PORTS = 4


processes = []
for i in range(TOTAL_NUM_PORTS):
    p = subprocess.Popen(["./launch_server.sh", str(STARTING_PORT + i)])
    processes.append(p)