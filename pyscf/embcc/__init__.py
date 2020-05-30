"""Embedded Coupled Cluster
Author: Max Nusspickel
"""

import logging
import os
from mpi4py import MPI

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

# === Logging

# EmbCCSD logging
loglevel = logging.DEBUG if __debug__ else logging.INFO
# Get unique log name
baselogname = "embcc.log"
logname = baselogname
idx = 0
while os.path.isfile(logname):
    idx += 1
    logname = baselogname + ".%d" % idx
MPI_comm.Barrier()
# Append MPI rank
if MPI_rank > 0:
    logname += ".mpi%d" % MPI_rank
logformat = "[{levelname:^8s}] {message:s}"
logging.basicConfig(level=loglevel, format=logformat, filename=logname, style="{")
#log = logging.getLogger(__name__)

import sys
# For PySCF output
sys.argv += ["-o", "pyscf.log"]

from .embcc import *
