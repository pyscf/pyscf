"""
This module implements the G0W0 approximation on top of `pyscf.tdscf.rhf_slow` TDHF implementation. Unlike `gw.py`, all
integrals are stored in memory. Several variants of GW are available:

 * `pyscf.gw_slow`: the molecular implementation;
 * (this module) `pyscf.pbc.gw.gw_slow`: single-kpoint PBC (periodic boundary condition) implementation;
 * `pyscf.pbc.gw.kgw_slow_supercell`: a supercell approach to PBC implementation with multiple k-points. Runs the
   molecular code for a model with several k-points for the cost of discarding momentum conservation and using dense
   instead of sparse matrixes;
 * `pyscf.pbc.gw.kgw_slow`: a PBC implementation with multiple k-points;
"""

# Convention for these modules:
# * IMDS contains routines for intermediates
# * kernel finds GW roots
# * GW provides a container

# This module and kgw_slow_supercell are simply aliases of the molecular code
from pyscf.gw import gw_slow


IMDS = gw_slow.IMDS
kernel = gw_slow.kernel
GW = gw_slow.GW
