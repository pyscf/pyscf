#!/usr/bin/env python

"""Load Gaussian, NWChem, and Molpro basis files and export NWChem data.

The parsers return PySCF internal basis data, which can be assigned to
mol.basis. NWChem ECP data can likewise be assigned to mol.ecp.
"""

import os
import tempfile
from pyscf import gto
from pyscf.gto.basis import parse_gaussian, parse_molpro, parse_nwchem

# Parsers can load files in other programs' formats. Use a Gaussian-format
# basis distributed with PySCF; load returns PySCF's internal basis data.
basis_dir = os.path.dirname(gto.basis.__file__)
gaussian_file = os.path.join(basis_dir, 'sapporo', 'sapporo-dzp_all.gbs')
sr_basis = parse_gaussian.load(gaussian_file, 'Sr')
print('Gaussian-format Sr shells =', len(sr_basis))

# NWChem basis and ECP data can also be converted back to text for export.
na_basis = parse_nwchem.load(os.path.join(basis_dir, 'ano.dat'), 'Na')
na_ecp = parse_nwchem.load_ecp(os.path.join(basis_dir, 'lanl2dz.dat'), 'Na')
print(parse_nwchem.convert_basis_to_nwchem('Na', na_basis))
print(parse_nwchem.convert_ecp_to_nwchem('Na', na_ecp))

# Molpro .libmol files can be loaded in the same way. The first line selects
# the element and shell; the second is the reference line in this file format.
with tempfile.TemporaryDirectory() as tmpdir:
    molpro_file = os.path.join(tmpdir, 'sample.libmol')
    with open(molpro_file, 'w') as f:
        f.write('C S : 2 1 1.2\n! Sample reference\n'
                ' 3.0 1.0 0.7 0.3\n')
    c_basis = parse_molpro.load(molpro_file, 'C')
print('Molpro-format C shells =', len(c_basis))
# See 30-read_molpro_orb.py for a Molpro orbital workflow.
