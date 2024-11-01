import subprocess, sys, os, signal
import threading

""" Run system commands with timeout
"""
class Command(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None
        self.out = None

    def run_command(self, capture = False):
        if not capture:
            self.process = subprocess.Popen(self.cmd,shell=True)
            self.process.communicate()
            return
        # capturing the outputs of shell commands
        self.process = subprocess.Popen(self.cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,stdin=subprocess.PIPE)
        out,err = self.process.communicate()
        if len(out) > 0:
            self.out = out.splitlines()
        else:
            self.out = None

    # set default timeout to 2 minutes
    def run(self, capture = False, timeout = 120):
        thread = threading.Thread(target=self.run_command, args=(capture,))
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            print('Command timeout, kill it: ' + self.cmd)
            self.process.terminate()
            thread.join()
        return self.out

if __name__ == "__main__":  
    '''basic test cases'''

    # run shell command without capture
    Command('pwd').run()
    # capture the output of a command
    date_time = Command('date').run(capture=True)
    print(date_time)
    fname = '../eisen/model_hp/eisen_1s_3p_new_dose_ed_AUC2_hpr0_nd_ex0_1s_0_temp_11.drh'
    # 'timeout', str(Timeout_dur), 
    cmd = "dReach -l 0 -k 2 -z {0} --precision {1}".format(fname, 0.01)
    # dReachCmd = Command(cmd).run(capture=True,timeout=5)
    # print(dReachCmd)
    Timeout_dur = 10
    st = ["dReach", "-k", str(2), "-z", fname, "--precision", str(0.01)]
    print(st)
    try:
        # output =  subprocess.run(st, capture_output = True, text = True, timeout=Timeout_dur).stdout
        # out = 0
        p = subprocess.Popen(st, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        # p_status = p.wait(timeout=Timeout_dur)
        output, err = p.communicate(timeout=Timeout_dur)
        # print('p_status', p_status)
        out = p.returncode
        print(out, output)
    except subprocess.TimeoutExpired as e:
        out = -15
        output = b'Timeout'
        print(f'Timeout ({Timeout_dur}s) expired', file=sys.stderr)
        # print('Terminating the whole process group...', file=sys.stderr)
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        #p.terminate()
    # kill a command after timeout
    ss = Command('ps aufxw|grep python').run(capture=True)
    print(ss)
    Command('echo "sleep 10 seconds"; sleep 10; echo "done"').run(timeout=2)

