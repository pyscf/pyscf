import os
import logging

import numpy as np
import scipy
import scipy.optimize

log = logging.getLogger(__name__)

__all__ = ["einsum", "time_string", "memory_string", "Options"]


def einsum(*args, **kwargs):
    kwargs["optimize"] = kwargs.pop("optimize", True)
    return np.einsum(*args, **kwargs)


def time_string(seconds, show_zeros=False):
    """String representation of seconds."""
    m, s = divmod(seconds, 60)
    if seconds >= 3600 or show_zeros:
        tstr = "%.0f h %.0f min %.0f s" % (divmod(m, 60) + (s,))
    elif seconds >= 60:
        tstr = "%.0f min %.1f s" % (m, s)
    else:
        tstr = "%.2f s" % s
    return tstr

get_time_string = time_string


def memory_string(nbytes, fmt='6.2f'):
    """String representation of nbytes"""
    if isinstance(nbytes, np.ndarray) and nbytes.size > 1:
        nbytes = nbytes.nbytes
    if nbytes < 1e3:
        val = nbytes
        mem = "B"
    elif nbytes < 1e6:
        val = nbytes / 1e3
        mem = "kB"
    elif nbytes < 1e9:
        val = nbytes / 1e6
        mem = "MB"
    elif nbytes < 1e12:
        val = nbytes / 1e9
        mem = "GB"
    else:
        val = nbytes / 1e12
        mem = "TB"
    return "{:{fmt}} {mem}".format(val, mem=mem, fmt=fmt)


class Options:
    def get(self, attr, default=None):
        if hasattr(self, attr):
            return getattr(self, attr)
        return default


def amplitudes_C2T(C1, C2):
    T1 = C1.copy()
    T2 = C2 - einsum("ia,jb->ijab", C1, C1)
    return T1, T2


def amplitudes_T2C(T1, T2):
    C1 = T1.copy()
    C2 = T2 + einsum("ia,jb->ijab", T1, T1)
    return C1, C2
