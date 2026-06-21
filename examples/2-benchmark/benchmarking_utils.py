import os
import pyscf
from pyscf.lib.logger import perf_counter, process_clock

def setup_logger():
    log = pyscf.lib.logger.Logger(verbose=5)
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if 'model name' in line:
                    log.note(line[:-1])
                    break
    except FileNotFoundError:
        pass
    try:
        with open('/proc/meminfo') as f:
            log.note(f.readline()[:-1])
    except FileNotFoundError:
        try:
            import psutil
            mem = psutil.virtual_memory()
            log.note(f'MemTotal: {mem.total // 1024} kB')
        except ImportError:
            pass
    log.note('OMP_NUM_THREADS=%s\n', os.environ.get('OMP_NUM_THREADS', None))
    return log

def get_cpu_timings():
    return process_clock(), perf_counter()
