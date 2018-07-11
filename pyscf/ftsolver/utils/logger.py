import sys
from datetime import datetime

stdout = sys.stdout
try:
    from mpi4py import MPI
    comm=MPI.COMM_WORLD
    rank=comm.Get_rank()
except:
    rank=0

def time():
    stdout.write(datetime.now().strftime("%y %b %d %H:%M:%S") + "  ")
    stdout.flush()

def result(msg, *args):
    if rank==0:
        time()
        stdout.write("********" + "  " + msg + "\n")
        stdout.flush()

def section(msg, *args):
    if rank==0:
        time()
        stdout.write("########" + "  " + msg + "\n")
        stdout.flush()

def debug(msg, *args):
    if rank==0:
        time()
        stdout.write("  DEBUG " + "  " + msg + "\n" )
        stdout.flush()

def info(msg, *args):
    if rank==0:
        time()
        stdout.write("  INFO  " + "  " + msg + "\n")
        stdout.flush()
def warning(msg, *args):
    if rank==0:
        time()
        stdout.write(" WARNING" + "  " + msg + "\n")
        stdout.flush()

def blank(msg, *args):
    if rank==0:
        stdout.write(msg + "\n")
        stdout.flush()

if __name__ == "__main__":
    msg = "This is a test"
    blank(msg)
    section(msg)
    debug(msg)
    info(msg)
           
