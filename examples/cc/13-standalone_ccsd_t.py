#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
CCSD(T) calculation requires only the CCSD t1,t2 amplitudes and relevant integrals.
One can run CCSD(T) calculation without recomputing HF and CCSD.
'''

from pyscf import gto, scf, lib
from pyscf import cc
from pyscf.cc import ccsd_t

mol = gto.M(atom=[('H', 0, 0, i) for i in range(4)],
            basis='ccpvdz')
#
# Run HF and CCSD and save results in file h10.chk
#
mf = scf.RHF(mol).set(chkfile='h10.chk').run()
mycc = cc.CCSD(mf).run()
lib.chkfile.save('h10.chk', 'cc/t1', mycc.t1)
lib.chkfile.save('h10.chk', 'cc/t2', mycc.t2)


#
# The following code can be executed in an independent script.  One can
# restore the old HF and CCSD results from chkfile then call a standalone
# CCSD(T) calculation.
#
mol = lib.chkfile.load_mol('h10.chk')
mf = scf.RHF(mol)
mf.__dict__.update(lib.chkfile.load('h10.chk', 'scf'))
mycc = cc.CCSD(mf)
mycc.__dict__.update(scf.chkfile.load('h10.chk', 'cc'))
eris = mycc.ao2mo()
ccsd_t.kernel(mycc, eris)

# Re-express the same restricted amplitudes in unrestricted spin blocks.
# No new SCF or CCSD optimization is needed for this representation change.
from pyscf.cc import uccsd_t, addons
ucc = cc.UCCSD(mf.to_uhf())
t1 = (mycc.t1, mycc.t1)
same_spin = mycc.t2 - mycc.t2.transpose(1, 0, 2, 3)
t2 = (same_spin, mycc.t2, same_spin)
uet = uccsd_t.kernel(ucc, ucc.ao2mo(), t1, t2)
# Conversion to GCCSD likewise preserves the reference and amplitudes.
gcc = addons.convert_to_gccsd(mycc)
get = gcc.ccsd_t()
print('Restricted, unrestricted, generalized triples corrections',
      ccsd_t.kernel(mycc, eris), uet, get)
