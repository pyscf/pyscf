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

def get_unique_name(basename):
    name = basename
    idx = 0
    while os.path.isfile(name):
        idx += 1
        name = basename + ".%d" % idx
    MPI_comm.Barrier()
    return name

logname = "embcc.log"
#logname = get_unique_name("embcc.log")
loglevel = logging.DEBUG if __debug__ else logging.INFO
# Append MPI rank
if MPI_rank > 0:
    logname += ".mpi%d" % MPI_rank
#logformat = "[{levelname:^8s}] {message:s}"
#logformat = "[{levelname:s}] {message:s}"
logformat = "{message:s}"
logging.basicConfig(level=loglevel, format=logformat, filename=logname, style="{")
log = logging.getLogger(__name__)
log.info("")
log.info("============================")
log.info("Importing module pyscf.embcc")
log.info("============================")
log.info("")

from .embcc import *
