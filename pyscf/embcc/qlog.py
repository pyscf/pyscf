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

Name            Level           Usage
----            -----           -----
CRITICAL        50              For immediate, non-recoverable errors
ERROR           40              For errors which are likely non-recoverable
WARNING         30              For possible errors and important information
INFO            20              Information, readable to users
INFOV   (*)     15  (-v)        Verbose information, readable to users
TIMING  (*)     12  (-vv)       Timing information for primary routines
DEBUG           10  (-vv)       Debugging information, indented for developers
DEBUGV  (*)      5  (-vvv)      Verbose debugging information
TIMINGV (*)      2  (-vvv)      Verbose timings information for secondary subroutines
"""

LVL_PREFIX = {
   "CRITICAL" : "CRITICAL",
   "ERROR" : "ERROR",
   "WARNING" : "WARNING",
   "DEBUGV" : "***",
   }

class QuantermFormatter(logging.Formatter):
    """Formatter which adds a prefix column and indentation."""

    def __init__(self, *args, prefix=True, prefix_width=10, prefix_sep='|',
            indent=False, indent_char=' ', indent_width=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix = prefix
        self.prefix_width = prefix_width
        self.prefix_sep = prefix_sep
        self.indent = indent
        self.indent_char = indent_char
        self.indent_width = indent_width

    def format(self, record):
        message = record.msg % record.args
        indent = prefix = ""
        if self.prefix:
            prefix = LVL_PREFIX.get(record.levelname, "")
            if prefix:
                prefix = "[%s]" % prefix
            prefix = "%-*s%s" % (self.prefix_width, prefix, self.prefix_sep)
        if self.indent:
            root = logging.getLogger()
            indent = root.indentLevel * self.indent_width * self.indent_char
        lines = [indent + x for x in message.split("\n")]
        lines = [((prefix + "  " + line) if line else prefix) for line in lines]
        return "\n".join(lines)


class QuantermFileHandler(logging.FileHandler):
    """Default file handler with IndentedFormatter"""


    def __init__(self, filename, mode='a', formatter=None, **kwargs):
        filename = get_logname(filename)
        super().__init__(filename, mode=mode, **kwargs)
        if formatter is None:
            formatter = QuantermFormatter()
        self.setFormatter(formatter)


def get_logname(basename, ext='log'):
    if ext:
        ext = '.' + ext
    name = '%s%s%s' % (basename, (('.mpi%d' % MPI_rank) if MPI_rank > 0 else ''), ext)
    return name


def init_logging():
    """Call this to initialize and configure logging, when importing the EmbCC module.

    This will:
    1) Add four new logging levels:
        `infov`, `timing`, `debugv`, and `timingv`.
    2) Adds the attribute `indentLevel` to the root logger and two new Logger methods:
        `setIndentLevel`, `changeIndentLevel`.
    """

    # Add new log levels
    # ------------------
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

    # Add indentation support
    # -----------------------
    # Note that indents are only tracked by the root logger
    root = logging.getLogger()
    root.indentLevel = 0

    def setIndentLevel(self, level):
        root = logging.getLogger()
        root.indentLevel = max(level, 0)
        return root.indentLevel

    def changeIndentLevel(self, delta):
        root = logging.getLogger()
        root.indentLevel = max(root.indentLevel + delta, 0)
        return root.indentLevel

    logging.Logger.setIndentLevel = setIndentLevel
    logging.Logger.changeIndentLevel = changeIndentLevel



#def init_default_logs(log, logname, loglevel):
#
#    logname = get_logname(logname)
#    warnname = get_logname("warnings")
#
#    log.setLevel(loglevel)
#    # Default log
#    fmt = EmbCCFormatter(prefix_sep='|', indent=True)
#    log.addHandler(EmbCCFileHandler(logname, formatter=fmt))
#    # Warning log (for WARNING and above)
#    wh = EmbCCFileHandler(warnname)
#    wh.setLevel(logging.WARNING)
#    log.addHandler(wh)
#
#    log.debugv("Created log file %s with level %d.", logname, loglevel)
