#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic
'''

import time
import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import parameters as param
from pyscf.lib import logger as log
from pyscf import scf
import pyscf.scf._vhf

class RHF:
    '''Non-relativistic restricted Hartree-Fock gradients'''
    def __init__(self, scf_method, restart=False):
        self.verbose = scf_method.verbose
        self.stdout = scf_method.stdout
        self.mol = scf_method.mol
        self.chkfile = scf_method.chkfile
        self.restart = restart
        self.scf = redo_scf(self.mol, scf_method)

    def dump_flags(self):
        log.info(self, '\n')
        log.info(self, '******** Gradients flags ********')
        log.info(self, 'potential = %s', self.get_coulomb_hf.__doc__)
        if self.restart:
            log.info(self, 'restart from chkfile  %s', self.chkfile)
        log.info(self, '\n')

    @lib.omnimethod
    def get_hcore(self, mol):
        h = mol.intor('cint1e_ipkin_sph', dim3=3) \
                + mol.intor('cint1e_ipnuc_sph', dim3=3)
        return h

    @lib.omnimethod
    def get_ovlp(self, mol):
        return mol.intor('cint1e_ipovlp_sph', dim3=3)

    def get_coulomb_hf(self, mol, dm):
        '''NR Hartree-Fock Coulomb repulsion'''
        log.info(self,'Compute Gradients of NR Hartree-Fock Coulomb repulsion')
        vj, vk = scf._vhf.direct_mapdm('cint2e_ip1_sph',  # (nabla i,j|k,l)
                                       'CVHFfill_dot_nrs2kl', # ip1_sph has k>=l,
# fill ij, ip1_sph has no-symm between i and j
                                       'CVHFunpack_nrblock2rect',
# funpack transposes (ij|kl) to (kl|ij), thus fvj,fvk ~ nr2sij_...
                                       ('CVHFnrs2ij_ij_s1kl', 'CVHFnrs2ij_il_s1jk'),
                                       dm, 3, # xyz, 3 components
                                       mol._atm, mol._bas, mol._env)
        vk = vk.transpose(0,2,1) # vk ~ vk_{jk} ~ {l, nabla i}, but vj ~ {nabla i, j}
        return vj - vk*.5

    @lib.omnimethod
    def calc_den_mat_e(self, scf_method):
        '''Energy weighted density matrix'''
        occ = scf_method.mo_occ
        mo0 = scf_method.mo_coeff[:,occ>0]
        e = scf_method.mo_energy
        mo0e = mo0 * e[occ>0] * occ[occ>0]
        return numpy.dot(mo0e, mo0.T.conj())

    def _grad_rinv(self, mol, ia):
        ''' for given atom, <|\\nabla r^{-1}|> '''
        mol.set_rinv_orig(mol.coord_of_atm(ia))
        return mol.charge_of_atm(ia) \
                * mol.intor('cint1e_iprinv_sph', dim3=3)

    def atom_of_aos(self, mol):
        return map(lambda s: s[0], mol.spheric_labels())

    def frac_atoms(self, mol, atm_id, mat):
        '''extract row band for each atom'''
        v = numpy.zeros_like(mat)
        blk = numpy.array(self.atom_of_aos(mol)) == atm_id
        v[:,blk,:] = mat[:,blk,:]
        return v

    def grad_e(self, mol, mf):
        t0 = (time.clock(), time.time())
        h1 = self.get_hcore(mol)
        s1 = self.get_ovlp(mol)
        dm0 = mf.calc_den_mat(mf.mo_coeff, mf.mo_occ)
        vhf = self.get_coulomb_hf(mol, dm0)
        log.timer(self, 'gradients of 2e part', *t0)
        f1 = h1 + vhf
        dme0 = self.calc_den_mat_e(mf)
        gs = []
        for ia in range(mol.natm):
            f = self.frac_atoms(mol, ia, f1) + self._grad_rinv(mol, ia)
            s = self.frac_atoms(mol, ia, s1)
            for i in range(3):
                v = lib.trace_ab(dm0, f[i]) - lib.trace_ab(dme0, s[i])
                gs.append(2 * v.real)
        gs = numpy.array(gs).reshape((mol.natm,3))
        log.debug(self, 'gradients of electronic part')
        log.debug(self, str(gs))
        return gs

    def grad_nuc(self, mol):
        gs = []
        for j in range(mol.natm):
            q2 = mol.charge_of_atm(j)
            r2 = mol.coord_of_atm(j)
            f = [0, 0, 0]
            for i in range(mol.natm):
                if i != j:
                    q1 = mol.charge_of_atm(i)
                    r1 = mol.coord_of_atm(i)
                    r = numpy.sqrt(numpy.dot(r1-r2,r1-r2))
                    for i in range(3):
                        f[i] += q1 * q2 * (r2[i] - r1[i])/ r**3
            gs.extend(f)
        gs = numpy.array(gs).reshape((mol.natm,3))
        log.debug(self, 'gradients of nuclear part')
        log.debug(self, str(gs))
        return gs

    def grad(self):
        cput0 = (time.clock(), time.time())
        if self.verbose >= param.VERBOSE_INFO:
            self.dump_flags()
        grads = self.grad_e(self.mol, self.scf) \
                + self.grad_nuc(self.mol)
        for ia in range(self.mol.natm):
            log.log(self, 'atom %d %s, force = (%.14g, %.14g, %.14g)', \
                    ia, self.mol.symbol_of_atm(ia), *grads[ia])
        log.timer(self, 'HF gradients', *cput0)
        return grads

def redo_scf(mol, mf):
    #if self.restart:
    #    log.info(self, 'Restart. Read HF results from chkfile.')
    #    mf.init_guess_by_chkfile(mol)
    if not mf.scf_conv:
        log.info(mf, 'SCF again before Gradients.')
        if mf.mo_coeff is not None:
            dm = mf.calc_den_mat(mf.mo_coeff, mf.mo_occ)
            scf_conv, hf_energy, mf.mo_energy, mf.mo_occ, mf.mo_coeff \
                    = mf.scf_cycle(mol, mf.conv_threshold*1e2, \
                                   init_dm=dm)
        else:
            scf_conv, hf_energy, mf.mo_energy, mf.mo_occ, mf.mo_coeff \
                    = mf.scf_cycle(mol, mf.conv_threshold*1e2)
    return mf


if __name__ == '__main__':
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [['He', (0.,0.,0.)], ]
    mol.basis = {'He': 'ccpvdz'}
    mol.build()
    method = scf.RHF(mol)
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
    g = RHF(rhf)
    print(g.grad())
#[[ 0   0                0             ]
# [ 0  -4.39690522e-03  -1.20567128e-02]
# [ 0   4.39690522e-03  -1.20567128e-02]]

