import platform
import subprocess
from contextlib import redirect_stdout

def evaluate(type: str, datadir: str, outfile: str) -> None:
    # parse files
    logfile = 'logs/' + outfile[:-4] + '.log'

    # on linux
    if platform.system() == 'Linux':
        from eval.evaluator import evaluate
        evaluate(type, datadir, outfile)

    # on kevin's mac
    elif platform.system() == 'Darwin':
    
        # Send file to Raspberry Pi
        cmd = "scp /Users/kevinhuestis/Development/AHLT/logs/{} pi@192.168.1.123:/home/pi/AHLT/".format(
            outfile, outfile)
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

    # on windows
    else:
        with open(logfile, 'a') as f:
            print('\n\nResults:\nTODO', file=f)

if __name__ == "__main__":
    evaluate('DDI', 'data/devel', 'DDI_2021-Jun-10-14:39.out')
    evaluate('NER', 'data/devel', 'NER_2021-Jun-10-14:33.out')