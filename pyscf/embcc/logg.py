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

"""
Log levels (* are non-standard):

name            level
----            -----
CRITICAL        50
ERROR           40
WARNING         30
INFO            20
INFOV   (*)     15      (active with -v)
TIMING  (*)     12      (active with -vv)
DEBUG           10      (active with -vv)
DEBUGV  (*)      5      (active with -vvv)
TIMINGV (*)      2      (active with -vvv)
"""

def init_logging(log, logname, loglevel):

    # Verbose INFO and DEBUG + TIMING
    def add_log_level(level, name):
        logging.addLevelName(level, name.upper())
        setattr(logging, name.upper(), level)
        def logForLevel(self, message, *args, **kwargs):
            if self.isEnabledFor(level):
                self._log(level, message, args, **kwargs)
        def logToRoot(message, *args, **kwargs):
            logging.log(level, message, *args, **kwargs)
        setattr(logging.getLoggerClass(), name, logForLevel)
        setattr(logging, name, logToRoot)
    add_log_level(15, "infov")
    add_log_level(12, "timing")
    add_log_level(5, "debugv")
    add_log_level(2, "timingv")

    def get_logname(basename):
        name = "%s%s.log" % (basename, ((".mpi%d" % MPI_rank) if MPI_rank > 0 else ""))
        return name

    logname = get_logname(logname)
    warnname = get_logname("warnings")

    # Note that indents are only tracked for the root logger
    rootlog = logging.getLogger("")
    rootlog.indentChar = " "
    rootlog.indentWidth = 4
    rootlog.indentLevel = 0

    def getIndent(self):
        indent = getattr(rootlog, "indentChar", " ") * getattr(rootlog, "indentLevel", 0) * getattr(rootlog, "indentWidth", 2)
        return indent

    logging.Logger.getIndent = getIndent

    def setIndentLevel(self, level):
        rootlog.indentLevel = max(level, 0)
        return rootlog.indentLevel

    def changeIndentLevel(self, delta):
        indent = getattr(rootlog, "indentLevel", 0)
        rootlog.indentLevel = max(indent + delta, 0)
        return rootlog.indentLevel

    logging.Logger.changeIndentLevel = changeIndentLevel

    #class IndentFormatter(logging.Formatter):
    #    """Formatter which adds indent of root logger."""
    #    def format(self, record):
    #        indent = rootlog.getIndent()
    #        msg = logging.Formatter.format(self, record)
    #        return "\n".join([indent + x for x in msg.split("\n")])

    lvl2prefix = {
            "DEBUGV" : "**",
            #"DEBUG" : "*",
            #"TIMING" : "T",
            "WARNING" : "WARNING",
            "ERROR" : "ERROR",
            "CRITICAL" : "CRITICAL"
            }

    class IndentedFormatter(logging.Formatter):

        def format(self, record):
            indent = rootlog.getIndent()
            message = record.msg % record.args
            prefix = lvl2prefix.get(record.levelname, "")
            if prefix:
                prefix = "[%s]" % prefix
            prefix = "%-10s|" % prefix
            lines = [indent + x for x in message.split("\n")]
            lines = [((prefix + "  " + line) if line else prefix) for line in lines]
            return "\n".join(lines)

    log.setLevel(loglevel)
    # Default log
    fh = logging.FileHandler(logname)
    #fh.setFormatter(IndentFormatter())
    fh.setFormatter(IndentedFormatter())
    log.addHandler(fh)
    # Warning log (for WARNING and above)
    wh = logging.FileHandler(warnname)
    #eh.setFormatter(Formatter())
    wh.setLevel(logging.WARNING)
    log.addHandler(wh)

    log.debugv("Created log file %s with level %d.", logname, loglevel)
