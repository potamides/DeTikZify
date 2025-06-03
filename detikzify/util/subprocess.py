import os
import signal
import subprocess

# Patched subprocess.run and subprocess.check_output that also kill children of
# the started process on timeouts (cf.
# https://alexandra-zaharia.github.io/posts/kill-subprocess-and-its-children-on-timeout-python/)
class _Popen(subprocess.Popen):
    def __init__(self, *args, **kwargs):
        if os.name == "nt":
            return super().__init__(
                *args, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, **kwargs
            )
        return super().__init__(*args, start_new_session=True, **kwargs)

    def safe_killpg(self):
        try:
            if os.name == "nt":
                # https://stackoverflow.com/a/28609523
                return os.kill(self.pid, signal.CTRL_BREAK_EVENT)
            return os.killpg(os.getpgid(self.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass # Supress the race condition error; bpo-40550.

def run(*popenargs, input=None, timeout=None, check=False, **kwargs):
    with _Popen(*popenargs, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(input, timeout=timeout)
        except subprocess.TimeoutExpired:
            process.safe_killpg()
            process.wait()
            raise
        except:
            process.safe_killpg()
            raise
        retcode = process.poll()
        if check and retcode:
            raise subprocess.CalledProcessError(
                retcode, process.args, output=stdout, stderr=stderr
            )
    return subprocess.CompletedProcess(
        process.args, retcode, stdout, stderr # type: ignore
    )

def check_output(*popenargs, timeout=None, **kwargs):
    return run(
        *popenargs, stdout=subprocess.PIPE, timeout=timeout, check=True, **kwargs
    ).stdout
