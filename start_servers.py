import subprocess
from constants import STARTING_PORT
from constants import TOTAL_NUM_PORTS


processes = []
p = subprocess.Popen(["./launch_server_main.sh", str(STARTING_PORT)])
for i in range(TOTAL_NUM_PORTS):
    p = subprocess.Popen(["./launch_server.sh", str(STARTING_PORT + i + 1)])
    processes.append(p)