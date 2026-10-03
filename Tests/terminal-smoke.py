"""Check the real Afterglow UI, owned native child failures, and terminal cleanup."""
import fcntl
import json
import os
import pty
import select
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time

binary = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else '.build/release/midnight-afterglow')

def read_until(master, marker, timeout=15):
    data = b''
    deadline = time.monotonic() + timeout
    while marker not in data and time.monotonic() < deadline:
        if select.select([master], [], [], .1)[0]:
            data += os.read(master, 65536)
    assert marker in data, data.decode(errors='replace')[-4000:]
    return data

with tempfile.TemporaryDirectory(prefix='afterglow-ui-') as root:
    source = os.path.join(root, 'empty source')
    output = os.path.join(root, 'new output')
    os.mkdir(source)
    for stop in (b'q', b'\x03'):
        master, slave = pty.openpty()
        original = termios.tcgetattr(slave)
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 30, 100, 0, 0))
        process = subprocess.Popen([binary], stdin=slave, stdout=slave, stderr=slave,
                                   env={**os.environ, 'TERM': 'xterm-256color'})
        try:
            data = read_until(master, b'midnight afterglow')
            if stop == b'q':
                # Exercise the actual self-launched native quantizer. Its source validation
                # must reject this empty folder without producing a completed artifact.
                os.write(master, b'2'); time.sleep(.1)
                os.write(master, b'\r' + source.encode() + b'\r'); time.sleep(.1)
                os.write(master, b'\t\r' + output.encode() + b'\r'); time.sleep(.1)
                os.write(master, b'r')
                data += read_until(master, b'Failed (exit', timeout=20)
                assert not os.path.exists(output), 'Failed validation created output'
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 5, 20, 0, 0))
            os.kill(process.pid, signal.SIGWINCH)
            os.write(master, stop)
            deadline = time.monotonic() + 10
            while process.poll() is None and time.monotonic() < deadline:
                if select.select([master], [], [], .1)[0]: data += os.read(master, 65536)
            assert process.poll() == 0, 'UI did not exit cleanly'
            while select.select([master], [], [], .1)[0]: data += os.read(master, 65536)
            assert b'\x1b[?1049l' in data, 'Alternate screen not restored'
            assert termios.tcgetattr(slave) == original, 'Terminal settings not restored'
            print('PTY passed:', 'native quantizer failure + quit' if stop == b'q' else 'Ctrl-C')
        finally:
            if process.poll() is None: process.kill(); process.wait()
            os.close(master); os.close(slave)
    result = subprocess.run([binary], capture_output=True, check=True)
    assert b'USAGE:' in result.stdout and b'\x1b' not in result.stdout
    print('Noninteractive help passed')
