from . import mp, ci, cc, mcscf, fci, tdscf
try:
    from . import doci
except ImportError:
    pass

# Note the agf2 module implicitly import mpi4py. This module should not be
# automatically imported until the dependency to mpi4py is completely removed.
