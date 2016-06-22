#import unittest 
import numpy as np

#from pyscf import gto, scf

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbchf
from pyscf.pbc.ao2mo import eris
import pyscf.pbc.tools

import ase
import ase.lattice
import ase.dft.kpoints

def make_cell(L, ngs):
    cell = pbcgto.Cell()
    cell.unit = 'B'
    #cell.atom.extend([['H', (0.,  0., 0.)], 
    #                  ['H', (1.4, 0., 0.)]])
    cell.atom.extend([['Be', (0.,  0., 0.)]])
    cell.h = L * np.identity(3)

    cell.basis = 'sto-3g'
    #cell.pseudo = None 
    cell.pseudo = 'gth-pade' 
    cell.gs = np.array([ngs,ngs,ngs])

    #cell.verbose = 4
    cell.build()
    return cell

#class KnowValues(unittest.TestCase):
#    def test_kpt_vs_supercell(self):
def test_kpt_vs_supercell():
        #############################################
        # Do a molecular calculation                #
        #############################################
        #mol = gto.Mole()
        #mol.unit = 'B' 
        #mol.atom.extend([['H', (0.,  0., 0.)], 
        #                 ['H', (1.4, 0., 0.)]])
        #mol.basis = 'sto-3g'
        #mol.build()
        #molmf = scf.RHF(mol)
        #molmf.verbose = 7
        #escf = molmf.scf()
        #vj, vk = molmf.get_jk(molmf.mol, molmf.make_rdm1())
        #ej = molmf.energy_elec(vhf=vj)[1]
        #print "E mol =", escf
        #print "J mol =", ej

        #############################################
        # Do a k-point sampling calculation         #
        #############################################

        L = 5.0
        ngs = 10
        cell = make_cell(L, ngs)
        print "cell gs =", cell.gs
        nk = (3, 1, 1)
        scaled_kpts = ase.dft.kpoints.monkhorst_pack(nk)
        abs_kpts = cell.get_abs_kpts(scaled_kpts)
        kmf = pbchf.KRHF(cell, abs_kpts, exxdiv=None)
        kmf.verbose = 7
        ekpt = kmf.scf()
        vj, vk = kmf.get_jk(kmf.cell, kmf.make_rdm1())
        ej_kpt = kmf.energy_elec(vhf_kpts=vj)[1]
        print "E kpt =", ekpt
        print "J kpt =", ej_kpt

        #############################################
        # Do a supercell Gamma-pt calculation       #
        #############################################

        supcell = pyscf.pbc.tools.super_cell(cell, nk)
        supcell.gs = np.array([nk[0]*ngs + (nk[0]-1)//2, 
                               nk[1]*ngs + (nk[1]-1)//2,
                               nk[2]*ngs + (nk[2]-1)//2])
        print "supcell gs =", supcell.gs
        #supcell.verbose = 7
        supcell.build()

        scaled_gamma = ase.dft.kpoints.monkhorst_pack((1,1,1))
        gamma = supcell.get_abs_kpts(scaled_gamma)
        mf = pbchf.KRHF(supcell, gamma, exxdiv=None)
        mf.verbose = 7
        esup = mf.scf()/np.prod(nk)
        vj, vk = mf.get_jk(mf.cell, mf.make_rdm1())
        ej_sup = mf.energy_elec(vhf_kpts=vj)[1]/np.prod(nk)
        print "E sup =", esup
        print "J sup =", ej_sup

        print "kpt sampling energy =", ekpt
        print "supercell energy    =", esup
        print "difference          =", ekpt-esup
        #self.assertAlmostEqual(ekpt, esup, 8)

        ###ecoul_kpt = kmf.energy_elec()[1]
        ###ecoul_sup = mf.energy_elec()[1]/np.prod(nk)
        ###print "kpt sampling ecoul =", ecoul_kpt 
        ###print "supercell ecoul    =", ecoul_sup

        kpts = abs_kpts
        nkpts = len(kpts)
        nocc = cell.nelectron // 2
        nmo = kmf.mo_coeff.shape[2]
        #print "nocc =", nocc
        #print "nmo =", nmo
        coul = 0 + 0j
        for ki in range(nkpts):
            for kj in range(nkpts):
                eris_kpt = eris.general(cell, 
                    [kmf.mo_coeff[ki,:,:,],kmf.mo_coeff[ki,:,:,],
                     kmf.mo_coeff[kj,:,:,],kmf.mo_coeff[kj,:,:,]],
                    [kpts[ki], kpts[ki], kpts[kj], kpts[kj]])
                #print eris_kpt
                for i in range(nocc):
                    for j in range(nocc):
                        coul += 2*eris_kpt[i*nmo+i,j*nmo+j]
        eri_j_kpt = coul.real/(np.prod(nk)**2)

        kpts = gamma
        nkpts = len(kpts)
        nocc = supcell.nelectron // 2
        nmo = mf.mo_coeff.shape[2]
        coul = 0 + 0j
        for ki in range(nkpts):
            for kj in range(nkpts):
                eris_kpt = eris.general(supcell, 
                    [mf.mo_coeff[ki,:,:,],mf.mo_coeff[ki,:,:,],
                     mf.mo_coeff[kj,:,:,],mf.mo_coeff[kj,:,:,]])
                for i in range(nocc):
                    for j in range(nocc):
                        coul += 2*eris_kpt[i*nmo+i,j*nmo+j]
        eri_j_sup = coul.real/np.prod(nk)

        print "J kpt =", ej_kpt
        print "J sup =", ej_sup
        print "ERI J kpt =", eri_j_kpt
        print "ERI J sup =", eri_j_sup

if __name__ == '__main__':
    #print("Full Tests for pbc.dft.krks")
    #unittest.main()
    test_kpt_vs_supercell()
