import pyscf
import pyscf.mcscf
import pyscf.fci
import numpy as np

'''
State average over different spin states

The mcscf.state_average_ function maybe not generate the right spin or spatial
symmetry as one needs.  One can modify the mc.fcisolver to handle arbitary
states.  The following code is based on pyscf/mcscf/addons.py state_average_
function
'''

# Mute warning msgs
pyscf.gto.mole.check_sanity = lambda *args: None

r = 1.8
mol = pyscf.gto.Mole()
mol.atom = [
    ['C', ( 0., 0.    , -r/2   )],
    ['C', ( 0., 0.    ,  r/2)],]
mol.basis = 'cc-pvdz'
mol.unit = 'B'
mol.symmetry = True
mol.build()
mf = pyscf.scf.RHF(mol)
mf.irrep_nelec = {'A1g': 4, 'E1gx': 0, 'E1gy': 0, 'A1u': 4,
                  'E1uy': 2, 'E1ux': 2, 'E2gx': 0, 'E2gy': 0, 'E2uy': 0, 'E2ux': 0}
ehf = mf.kernel()
#mf.analyze()

# state-average over 1 triplet + 2 singlets
weights = np.ones(3)/3
fcibase = pyscf.fci.direct_spin1_symm.FCISolver(mol)
fcibase_class = fcibase.__class__
solver1 = pyscf.fci.addons.fix_spin(fcibase, shift=.2, ss_value=2)
solver1.nroots = 1
solver2 = pyscf.fci.addons.fix_spin(fcibase, ss_value=0)
solver2.nroots = 2
class FakeCISolver(fcibase_class):
    def kernel(self, h1, h2, ncas, nelecas, ci0=None, **kwargs):
# Note self.orbsym is initialized lazily in mc1step_symm.kernel function
        e1, c1 = solver1.kernel(h1, h2, ncas, nelecas, ci0, orbsym=self.orbsym)
        e2, c2 = solver2.kernel(h1, h2, ncas, nelecas, ci0, orbsym=self.orbsym)
        e = [e1, e2[0], e2[1]]
        c = [c1, c2[0], c2[1]]
        for i, ei in enumerate(e):
            ss = pyscf.fci.spin_op.spin_square0(c[i], ncas, nelecas)
            pyscf.lib.logger.info(mc, 'state %d  E = %.15g S^2 = %.7f',
                                  i, ei, ss[0])
        return np.einsum('i,i', np.array(e), weights), c
    def approx_kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
        e1, c1 = solver1.kernel(h1, h2, norb, nelec, ci0, orbsym=self.orbsym,
                                max_cycle=mc.ci_response_space)
        e2, c2 = solver2.kernel(h1, h2, norb, nelec, ci0, orbsym=self.orbsym,
                                max_cycle=mc.ci_response_space)
        e = [e1, e2[0], e2[1]]
        c = [c1, c2[0], c2[1]]
        return np.einsum('i,i->', e, weights), c
    def make_rdm1(self, ci0, norb, nelec):
        dm1 = 0
        for i, wi in enumerate(weights):
            dm1 += wi*fcibase_class.make_rdm1(self, ci0[i], norb, nelec)
        return dm1
    def make_rdm12(self, ci0, norb, nelec):
        rdm1 = 0
        rdm2 = 0
        for i, wi in enumerate(weights):
            dm1, dm2 = fcibase_class.make_rdm12(self, ci0[i], norb, nelec)
            rdm1 += wi * dm1
            rdm2 += wi * dm2
        return rdm1, rdm2
    def spin_square(self, ci0, norb, nelec):
        ss = [pyscf.fci.spin_op.spin_square0(x, norb, nelec)[0] for x in ci0]
        ss = np.einsum('i,i->', weights, ss)
        multip = np.sqrt(ss+.25)*2
        return ss, multip

mc = pyscf.mcscf.CASSCF(mf, 8, 8)
mc.verbose = 4
mc.fcisolver = FakeCISolver()
mc.kernel()
mo = mc.mo_coeff
