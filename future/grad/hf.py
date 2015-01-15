#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic
'''

import time
import numpy
import pyscf.lib
import pyscf.lib.logger as log
import pyscf.lib.parameters as param
from pyscf.scf import _vhf


def grad_elec(mfg, mo_energy=None, mo_occ=None, mo_coeff=None):
    t0 = (time.clock(), time.time())
    mf = mfg._scf
    mol = mfg.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    h1 = mfg.get_hcore(mol)
    s1 = mfg.get_ovlp(mol)
    dm0 = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    vhf = mfg.get_veff(mol, dm0)
    log.timer(mfg, 'gradients of 2e part', *t0)
    f1 = h1 + vhf
    dme0 = mfg.make_rdm1e(mf.mo_energy, mf.mo_occ, mf.mo_coeff)
    gs = []
    for ia in range(mol.natm):
        f = mfg.matblock_by_atom(mol, ia, f1) + mfg._grad_rinv(mol, ia)
        s = mfg.matblock_by_atom(mol, ia, s1)
        for i in range(3):
            v = numpy.einsum('ij,ji', dm0, f[i]) \
              - numpy.einsum('ij,ji', dme0, s[i])
            gs.append(2 * v.real)
    gs = numpy.array(gs).reshape((mol.natm,3))
    log.debug(mfg, 'gradients of electronic part')
    log.debug(mfg, str(gs))
    return gs

def grad_nuc(mol):
    gs = []
    for j in range(mol.natm):
        q2 = mol.atom_charge(j)
        r2 = mol.atom_coord(j)
        f = [0, 0, 0]
        for i in range(mol.natm):
            if i != j:
                q1 = mol.atom_charge(i)
                r1 = mol.atom_coord(i)
                r = numpy.sqrt(numpy.dot(r1-r2,r1-r2))
                for i in range(3):
                    f[i] += q1 * q2 * (r2[i] - r1[i])/ r**3
        gs.extend(f)
    gs = numpy.array(gs).reshape((mol.natm,3))
    log.debug(mol, 'gradients of nuclear part')
    log.debug(mol, str(gs))
    return gs


def get_hcore(mol):
        h = mol.intor('cint1e_ipkin_sph', comp=3) \
                + mol.intor('cint1e_ipnuc_sph', comp=3)
        return h

def get_ovlp(mol):
    return mol.intor('cint1e_ipovlp_sph', comp=3)

def get_veff(mol, dm):
    return get_coulomb_hf(mol, dm)
def get_coulomb_hf(mol, dm):
    '''NR Hartree-Fock Coulomb repulsion'''
    log.info(mol, 'Compute Gradients of NR Hartree-Fock Coulomb repulsion')
    #vj, vk = pyscf.scf.hf.get_vj_vk(pycint.nr_vhf_grad_o1, mol, dm)
    #return vj - vk*.5
    vj, vk = _vhf.direct_mapdm('cint2e_ip1_sph',  # (nabla i,j|k,l)
                               's2kl', # ip1_sph has k>=l,
                               ('kl->s1ij', 'kj->s1il'),
                               dm, 3, # xyz, 3 components
                               mol._atm, mol._bas, mol._env)
    return vj - vk*.5

def make_rdm1e(mo_energy, mo_occ, mo_coeff):
    '''Energy weighted density matrix'''
    mo0 = mo_coeff[:,mo_occ>0]
    mo0e = mo0 * mo_energy[mo_occ>0] * mo_occ[mo_occ>0]
    return numpy.dot(mo0e, mo0.T.conj())

def matblock_by_atom(mol, atm_id, mat):
    '''extract row band for each atom'''
    shells = mol.atom_shell_ids(atm_id)
    b0, b1 = mol.nao_nr_range(shells[0], shells[-1]+1)
    v = numpy.zeros_like(mat)
    v[:,b0:b1,:] = mat[:,b0:b1,:]
    return v


class RHF:
    '''Non-relativistic restricted Hartree-Fock gradients'''
    def __init__(self, scf_method):
        self.verbose = scf_method.verbose
        self.stdout = scf_method.stdout
        self.mol = scf_method.mol
        self._scf = scf_method
        self.chkfile = scf_method.chkfile

    def dump_flags(self):
        pass
#        log.info(self, '\n')
#        log.info(self, '******** Gradients flags ********')
#        if not self._scf.converged:
#            log.warn(self, 'underneath SCF of gradients not converged')
#        log.info(self, '\n')

    @pyscf.lib.omnimethod
    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        return get_hcore(mol)

    @pyscf.lib.omnimethod
    def get_ovlp(self, mol=None):
        if mol is None:
            mol = self.mol
        return get_ovlp(mol)

    def get_veff(self, mol, dm):
        return _hack_mol_log(mol, self, get_coulomb_hf, dm)

    def make_rdm1e(self, mo_energy, mo_occ, mo_coeff):
        return make_rdm1e(mo_energy, mo_occ, mo_coeff)

    def _grad_rinv(self, mol, ia):
        ''' for given atom, <|\\nabla r^{-1}|> '''
        mol.set_rinv_orig_(mol.atom_coord(ia))
        return mol.atom_charge(ia) * mol.intor('cint1e_iprinv_sph', comp=3)

    def matblock_by_atom(self, mol, atm_id, mat):
        return matblock_by_atom(mol, atm_id, mat)

    def grad_elec(self, mo_energy=None, mo_occ=None, mo_coeff=None):
        return grad_elec(self, mo_energy, mo_occ, mo_coeff)

    def grad_nuc(self, mol=None):
        if mol is None: mol = self.mol
        return _hack_mol_log(mol, self, grad_nuc)

    def grad(self, mo_energy=None, mo_occ=None, mo_coeff=None):
        cput0 = (time.clock(), time.time())
        if self.verbose >= param.VERBOSE_INFO:
            self.dump_flags()
        grads = self.grad_elec(mo_energy, mo_occ, mo_coeff) + self.grad_nuc()
        for ia in range(self.mol.natm):
            log.log(self, 'atom %d %s, force = (%.14g, %.14g, %.14g)', \
                    ia, self.mol.atom_symbol(ia), *grads[ia])
        log.timer(self, 'HF gradients', *cput0)
        return grads


def _hack_mol_log(mol, dev, fn, *args, **kwargs):
    verbose_bak, mol.verbose = mol.verbose, dev.verbose
    stdout_bak,  mol.stdout  = mol.stdout , dev.stdout
    res = fn(mol, *args, **kwargs)
    mol.verbose = verbose_bak
    mol.stdout  = stdout_bak
    return res

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [['He', (0.,0.,0.)], ]
    mol.basis = {'He': 'ccpvdz'}
    mol.build()
    method = scf.RHF(mol)
    method.scf()
    g = RHF(method)
    print(g.grad())

    h2o = gto.Mole()
    h2o.verbose = 0
    h2o.output = None#'out_h2o'
    h2o.atom = [
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]
    h2o.basis = {'H': '631g',
                 'O': '631g',}
    h2o.build()
    rhf = scf.RHF(h2o)
    rhf.scf()
    g = RHF(rhf)
    print(g.grad())
#[[ 0   0                0             ]
# [ 0  -4.39690522e-03  -1.20567128e-02]
# [ 0   4.39690522e-03  -1.20567128e-02]]

