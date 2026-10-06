#!/usr/bin/env python

'''
Spin-orbit coupling (SOC) with GTH pseudopotentials

This involves two settings:
1. Select a GTH pseudopotential, including the string "SOC" in the name of
pseudo to indicate the use of SOC pseudopotential.
2. Set with_soc=True for GHF/GKS to include the SOC contribution.

SOC and scalar-only GTH potentials can be mixed. Atoms whose potentials
contain **SOC** contribute SOC projectors.
'''

import pyscf

cell = pyscf.M(
    a='''
        0.0 3.0 3.0
        3.0 0.0 3.0
        3.0 3.0 0.0
    ''',
    atom='''
        Pb 0.0 0.0 0.0
        S  3.0 3.0 3.0
    ''',
    unit='Angstrom',
    basis={
        'Pb': 'DZVP-MOLOPT-PBE-GTH-q4',
        'S': 'DZVP-MOLOPT-PBE-GTH-q6',
    },
    # SOC and scalar-only GTH potentials can be mixed.
    # If with_soc=True is specified, scalar PP contributes zeros potentials.
    pseudo={
        'Pb': 'GTH-SOC-PBE-q4',
        'S': 'GTH-PBE-q6',  # Scalar-only sulfur; SOC projectors come from Pb.
    },
)

kmesh = [2, 2, 2]
kpts = cell.make_kpts(kmesh)
mf = cell.KGKS(xc='PBE', kpts=kpts)
mf.collinear = 'mcol'
mf.with_soc = True
mf.kernel()
