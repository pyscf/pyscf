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

def get_unique_name(basename):
    name = basename
    idx = 0
    while os.path.isfile(name):
        idx += 1
        name = basename + ".%d" % idx
    if MPI: MPI_comm.Barrier()
    return name

logname = "embcc.log"
#logname = get_unique_name("embcc.log")
loglevel = logging.DEBUG if __debug__ else logging.INFO
# Append MPI rank
if MPI_rank > 0:
    logname += ".mpi%d" % MPI_rank
#logformat = "[{levelname:^8s}] {message:s}"
#logformat = "[{levelname:s}] {message:s}"
#logformat = "{message:s}"
#logging.basicConfig(level=loglevel, format=logformat, filename=logname, style="{")
#log = logging.getLogger(__name__)
rootlog = logging.getLogger("")
rootlog.indentChar = " "
rootlog.indentWidth = 4
rootlog.indentLevel = 0

def getIndent(self):
    #indent = getattr(self, "indentChar", " ") * getattr(self, "indentLevel", 0) * getattr(self, "indentWidth", 2)
    indent = getattr(rootlog, "indentChar", " ") * getattr(rootlog, "indentLevel", 0) * getattr(rootlog, "indentWidth", 2)
    return indent

def changeIndentLevel(self, delta):
    #indent = getattr(self, "indentLevel", 0)
    indent = getattr(rootlog, "indentLevel", 0)
    #self.indentLevel = max(indent + delta, 0)
    rootlog.indentLevel = max(indent + delta, 0)
    #return self.indentLevel
    return rootlog.indentLevel

logging.Logger.getIndent = getIndent
logging.Logger.changeIndentLevel = changeIndentLevel

class Formatter(logging.Formatter):

    def format(self, record):
        #log = logging.getLogger(record.name)
        #indent = log.getIndent()
        indent = rootlog.getIndent()
        msg = logging.Formatter.format(self, record)
        return "\n".join([indent + x for x in msg.split("\n")])

log = logging.getLogger(__name__)
log.setLevel(loglevel)
#log.indentLevel = 0
#log.indentChar = "."
fh = logging.FileHandler(logname)
fh.setFormatter(Formatter())
log.addHandler(fh)

log.info("+------------------------------+")
log.info("| Importing module pyscf.embcc |")
log.info("+------------------------------+")
log.info("  Author: Max Nusspickel")
log.info("  Email: max.nusspickel@gmail.com")
log.info("")

from .embcc import *
