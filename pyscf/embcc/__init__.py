"""Embedded Coupled Cluster
Author: Max Nusspickel
"""

import logging
import os

try:
    from mpi4py import MPI
    MPI_comm = MPI.COMM_WORLD
    MPI_rank = MPI_comm.Get_rank()
    MPI_size = MPI_comm.Get_size()
except ModuleNotFoundError:
    MPI = False
    MPI_rank = 0
    MPI_size = 1

# Logging
# =======
def get_logname(basename):
    name = "%s%s.log" % (basename, ((".mpi%d" % MPI_rank) if MPI_rank > 0 else ""))
    return name

LOGNAME = get_logname("embcc")
ERRNAME = get_logname("warnings")
LOGLEVEL = logging.DEBUG if __debug__ else logging.INFO

# Note that indents are only tracked for the root logger
rootlog = logging.getLogger("")
rootlog.indentChar = " "
rootlog.indentWidth = 4
rootlog.indentLevel = 0

def getIndent(self):
    indent = getattr(rootlog, "indentChar", " ") * getattr(rootlog, "indentLevel", 0) * getattr(rootlog, "indentWidth", 2)
    return indent

logging.Logger.getIndent = getIndent

def changeIndentLevel(self, delta):
    indent = getattr(rootlog, "indentLevel", 0)
    rootlog.indentLevel = max(indent + delta, 0)
    return rootlog.indentLevel

logging.Logger.changeIndentLevel = changeIndentLevel

class IndentFormatter(logging.Formatter):
    """Formatter which adds indent of root logger."""
    def format(self, record):
        indent = rootlog.getIndent()
        msg = logging.Formatter.format(self, record)
        return "\n".join([indent + x for x in msg.split("\n")])

log = logging.getLogger(__name__)
log.setLevel(LOGLEVEL)
# Default log
fh = logging.FileHandler(LOGNAME)
fh.setFormatter(IndentFormatter())
log.addHandler(fh)
# Error log (for WARNING and above)
eh = logging.FileHandler(ERRNAME)
#eh.setFormatter(Formatter())
eh.setLevel(logging.WARNING)
log.addHandler(eh)

log.info("+--------------------+")
log.info("| Module pyscf.embcc |")
log.info("+--------------------+")
log.info("  Author: Max Nusspickel")
log.info("  Email: max.nusspickel@gmail.com")
log.info("")

from .embcc import EmbCC
