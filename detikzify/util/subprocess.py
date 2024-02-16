from os import killpg, getpgid
from subprocess import Popen, TimeoutExpired, CalledProcessError, CompletedProcess, PIPE
from signal import SIGKILL

# Patch subprocess.run and subprocess.check_output to also kill children of the
# started process on timeouts (cf.
# https://alexandra-zaharia.github.io/posts/kill-subprocess-and-its-children-on-timeout-python/)
def run(*popenargs, input=None, timeout=None, check=False, **kwargs):
    with Popen(*popenargs, start_new_session=True, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(input, timeout=timeout)
        except TimeoutExpired:
            killpg(getpgid(process.pid), SIGKILL)
            process.wait()
            raise
        except:
            killpg(getpgid(process.pid), SIGKILL)
            raise
        retcode = process.poll()
        if check and retcode:
            raise CalledProcessError(retcode, process.args,
                                     output=stdout, stderr=stderr)
    return CompletedProcess(process.args, retcode, stdout, stderr) # type: ignore

def check_output(*popenargs, timeout=None, **kwargs):
    return run(*popenargs, stdout=PIPE, timeout=timeout, check=True, **kwargs).stdout
