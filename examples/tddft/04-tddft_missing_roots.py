#!/usr/bin/env python
#
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#


from pyscf import gto, scf, tdscf


''' TDSCF selected particle-hole pairs that have low energy difference (e.g., HOMO -> LUMO,
    HOMO -> LUMO+1 etc.) as init guess for the iterative solution of the TDSCF equation.
    In cases where there are many near degenerate p-h pairs, their contributions to the
    final TDSCF solutions can all be significnat and all such pairs should be included.

    The determination of such near degeneracies in p-h pairs is controlled by the
    `deg_eia_thresh` variable: if the excitation energy of two p-h pairs differs less
    than `deg_eia_thresh`, they will be considered degenerate and selected simultaneously
    as init guess.

    This example demonstrates the use of this variable to reduce the chance of missing
    roots in TDSCF calculations. A molecule is used here for demonstration but the same
    applies to periodic TDSCF calculations.
'''

atom = '''
O          0.00000        0.00000        0.11779
H          0.00000        0.75545       -0.47116
H          0.00000       -0.75545       -0.47116
O          4.00000        0.00000        0.11779
H          4.00000        0.75545       -0.47116
H          4.00000       -0.75545       -0.47116
'''
basis = 'cc-pvdz'

mol = gto.M(atom=atom, basis=basis)
mf = scf.RHF(mol).density_fit().run()

''' By default, `deg_eia_thresh` = 1e-3 (Ha) as can be seen in the output if verbose >= 4.
'''
mf.TDA().set(nroots=6).run()

''' One can check if there are missing roots by using a larger value for `deg_eia_thresh`.
    Note that this will increase the number of init guess vectors and increase the cost of
    solving the TDSCF equation.
'''
mf.TDA().set(nroots=6, deg_eia_thresh=0.1).run()
