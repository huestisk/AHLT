import sys
import subprocess

# parse arguments
datadir = sys.argv[1]
outfile = sys.argv[2]

# Send file to Raspberry Pi
cmd = "scp /Users/kevinhuestis/Development/AHLT/{} pi@192.168.1.123:/home/pi/AHLT/{}".format(
    outfile, outfile)
subprocess.Popen([cmd], shell=True)

""" Connect to Raspberry pi for eval """
# Note: Pub key must be added to raspberry pi
user = 'pi'
host = '192.168.1.123'
cmd = "python3 AHLT/evaluator.pyc DDI AHLT/{} AHLT/{}".format(datadir, outfile)
ssh_cmd = 'ssh %s@%s "%s"' % (user, host, cmd)
p = subprocess.Popen([ssh_cmd], shell=True)
