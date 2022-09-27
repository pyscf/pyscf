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

