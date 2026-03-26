#!/usr/bin/env python

'''
SO-ECP contributions in SCF calculations under PBCs are supported for GHF and
GKS methods. This example demonstrates how to enable the SOC part in a periodic
GHF calculation.

See also examples/scf/44-soc_ecp.py
'''

import numpy as np
import pyscf
cell = pyscf.M(
    a = np.eye(3) * 5,
    atom = 'C 0 0 0; C 0 0 1.5',
    basis = 'crenbl',
    ecp = {'C': 'crenbl'} # crenbl and crenbs ECP includes SOC data
)

mf = cell.GHF().density_fit()
# For GHF (and KGHF/GKS) methods, the spin-orbit part of the ECP is
# disabled by default. It can be explicitly enabled by setting
# the `with_soc` attribute.
mf.with_soc = True
mf.kernel()

# The SOC term from the ECP is a 2N x 2N matrix, corresponding to the integrals
#     <|1j * s · U_SOC|> 
# This matrix can be obtained using the `ecp_int` function.
from pyscf.pbc.gto.ecp import ecp_int
h_soc = ecp_int(cell, kpt, intor='ECPso')
