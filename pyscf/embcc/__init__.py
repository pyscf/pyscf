"""Embedded Coupled Cluster
Author: Max Nusspickel
Email:  max.nusspickel@gmail.com
"""

import os.path
import logging
from .logg import init_logging

log = logging.getLogger(__name__)
init_logging(log)

log.info("+--------------------+")
log.info("| Module pyscf.embcc |")
log.info("+--------------------+")
log.info("  Author: Max Nusspickel")
log.info("  Email:  max.nusspickel@gmail.com")
log.info("")

# Required modules
log.debug("Required modules:")
log.changeIndentLevel(1)
try:
    import numpy
    log.debug("NumPy  v%-8s  found at   %s", numpy.__version__, os.path.dirname(numpy.__file__))
except ImportError:
    log.critical("NumPy not found.")
    raise
try:
    import scipy
    log.debug("SciPy  v%-8s  found at   %s", scipy.__version__, os.path.dirname(scipy.__file__))
except ImportError:
    log.critical("SciPy not found.")
    raise
try:
    import h5py
    log.debug("h5py   v%-8s  found at   %s", h5py.__version__, os.path.dirname(h5py.__file__))
except ImportError:
    log.critical("h5py not found.")
    raise
try:
    import mpi4py
    log.debug("mpi4py v%-8s  found at   %s", mpi4py.__version__, os.path.dirname(mpi4py.__file__))
    from mpi4py import MPI
    MPI_comm = MPI.COMM_WORLD
    MPI_rank = MPI_comm.Get_rank()
    MPI_size = MPI_comm.Get_size()
    log.debug("MPI(rank= %d  size= %d)", MPI_rank, MPI_size)
except ImportError:
    log.debug("mpi4py not found.")
log.changeIndentLevel(-1)
log.debug("")

from .embcc import EmbCC
