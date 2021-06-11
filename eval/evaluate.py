import os
import platform
import subprocess
from contextlib import redirect_stdout

cwd = os.getcwd()

def evaluate(type: str, datadir: str, outfile: str) -> None:
    # parse files
    logfile = 'logs/' + outfile[:-4] + '.log'
    os = platform.system()

    # on linux
    if os == 'Linux':
        from eval.evaluator import evaluate
        evaluate(type, datadir, outfile)

    # on kevin's mac or window's
    elif os == 'Darwin':
    
        # Send file to Raspberry Pi
        cmd = "scp " + cwd + "/logs/{} pi@192.168.1.123:/home/pi/AHLT/".format(outfile, outfile)        
        subprocess.Popen([cmd], shell=True)

        """ Connect to Raspberry pi for eval """
        # Note: Pub key must be added to raspberry pi
        user = 'pi'
        host = '192.168.1.123'
        cmd = "python3 AHLT/evaluator.pyc {} AHLT/{} AHLT/{}".format(type, datadir, outfile)
        ssh_cmd = 'ssh %s@%s "%s"' % (user, host, cmd)

        with open(logfile, 'a') as f:
            print('\n\nResults:', file=f)
            p = subprocess.Popen([ssh_cmd], shell=True, stdout=f, stderr=f)

    elif os == 'Windows':
        with open(logfile, 'a') as f:
            print('\n\nResults:\nTODO', file=f)

if __name__ == "__main__":
    evaluate('DDI', 'data/devel', 'DDI_2021-Jun-11-1523.out')
    # evaluate('NER', 'data/devel', 'NER_2021-Jun-11-1038.out')