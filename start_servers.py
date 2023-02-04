import subprocess
from subprocess import Popen


TOTAL_NUM_PORTS = 4

processes = []
for i in range(TOTAL_NUM_PORTS):
    p = subprocess.Popen(["./launch_server.sh", str(10000 + i)])
    processes.append(p)