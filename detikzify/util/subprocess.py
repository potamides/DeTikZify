from os import killpg, getpgid
from subprocess import Popen, TimeoutExpired, CalledProcessError, CompletedProcess, PIPE
from signal import SIGKILL

def safe_killpg(pid, signal):
    try:
        killpg(getpgid(pid), signal)
    except ProcessLookupError:
        pass # Supress the race condition error; bpo-40550.

# Patch subprocess.run and subprocess.check_output to also kill children of the
# started process on timeouts (cf.
# https://alexandra-zaharia.github.io/posts/kill-subprocess-and-its-children-on-timeout-python/)
def run(*popenargs, input=None, timeout=None, check=False, **kwargs):
    with Popen(*popenargs, start_new_session=True, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(input, timeout=timeout)
        except TimeoutExpired:
            safe_killpg(process.pid, SIGKILL)
            process.wait()
            raise
        except:
            safe_killpg(process.pid, SIGKILL)
            raise
        retcode = process.poll()
        if check and retcode:
            raise CalledProcessError(retcode, process.args,
                                     output=stdout, stderr=stderr)
    return CompletedProcess(process.args, retcode, stdout, stderr) # type: ignore

def check_output(*popenargs, timeout=None, **kwargs):
    return run(*popenargs, stdout=PIPE, timeout=timeout, check=True, **kwargs).stdout
