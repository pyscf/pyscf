'''
Average of configuration Hartree-Fock
as described in DIRAC by Timo Fleig
current plan is to use this to generate spherical symmetric reference atom.
'''

from functools import reduce
import numpy
import pyscf.gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf, rohf, uhf
import pyscf.scf.chkfile
from pyscf import __config__

def get_occ(mf, mo_energy=None, mo_coeff=None, nclose=None, nact=None, nopen=None):
    '''Label occupancies for each orbitals

    Kwargs:
        mo_energy : 1D ndarray
            Obital energies

        mo_coeff : 2D ndarray
            Obital coefficients
        
        nclose : int
            Number of core orbitals

        nact : int
            Number of active electrons

        nopen : int
            Number of active orbitals
    '''
    if mo_energy is None: mo_energy = mf.mo_energy
    if nclose is None : nclose = mf.nclose
    if nact is None : nact = mf.nact
    if nopen is None : nopen = mf.nopen

    if mo_energy is None: mo_energy = mf.mo_energy
    e_idx_a = numpy.argsort(mo_energy[0])
    e_idx_b = numpy.argsort(mo_energy[1])
    e_sort_a = mo_energy[0][e_idx_a]
    e_sort_b = mo_energy[1][e_idx_b]
    nmo = mo_energy[0].size
    n_a, n_b = nclose+nopen, nclose+nopen
    mo_occ = numpy.zeros_like(mo_energy)
    mo_occ[0,e_idx_a[:nclose]] = 1
    mo_occ[1,e_idx_b[:nclose]] = 1
    mo_occ[0,e_idx_a[nclose:nclose+nopen]] = nact/nopen/2.
    mo_occ[1,e_idx_b[nclose:nclose+nopen]] = nact/nopen/2.
    if mf.verbose >= logger.INFO and n_a < nmo and n_b > 0 and n_b < nmo:
        if e_sort_a[n_a-1]+1e-3 > e_sort_a[n_a]:
            logger.warn(mf, 'alpha nocc = %d  HOMO %.15g >= LUMO %.15g',
                        n_a, e_sort_a[n_a-1], e_sort_a[n_a])
        else:
            logger.info(mf, '  alpha nocc = %d  HOMO = %.15g  LUMO = %.15g',
                        n_a, e_sort_a[n_a-1], e_sort_a[n_a])

        if e_sort_b[n_b-1]+1e-3 > e_sort_b[n_b]:
            logger.warn(mf, 'beta  nocc = %d  HOMO %.15g >= LUMO %.15g',
                        n_b, e_sort_b[n_b-1], e_sort_b[n_b])
        else:
            logger.info(mf, '  beta  nocc = %d  HOMO = %.15g  LUMO = %.15g',
                        n_b, e_sort_b[n_b-1], e_sort_b[n_b])

        if e_sort_a[n_a-1]+1e-3 > e_sort_b[n_b]:
            logger.warn(mf, 'system HOMO %.15g >= system LUMO %.15g',
                        e_sort_b[n_a-1], e_sort_b[n_b])

        numpy.set_printoptions(threshold=nmo)
        logger.debug(mf, '  alpha mo_energy =\n%s', mo_energy[0])
        logger.debug(mf, '  beta  mo_energy =\n%s', mo_energy[1])
        numpy.set_printoptions(threshold=1000)

    if mo_coeff is not None and mf.verbose >= logger.DEBUG:
        ss, s = mf.spin_square((mo_coeff[0][:,mo_occ[0]>0],
                                mo_coeff[1][:,mo_occ[1]>0]), mf.get_ovlp())
        logger.debug(mf, 'multiplicity <S^2> = %.8g  2S+1 = %.8g', ss, s)
    return mo_occ

def make_rdm1(mo_coeff, mo_occ, **kwargs):
    '''One-particle densit matrix.  mo_occ is a 1D array, with occupancy 1 or 2.
    '''
    print(mo_occ)
    mo_a = mo_coeff[:,mo_occ>0]
    mo_b = mo_coeff[:,mo_occ==2]
    occ_a = numpy.zeros(mo_occ.size)
    occ_a[mo_occ==2] = 1
    occ_a = mo_occ-occ_a
    print(occ_a)
    dm_a = numpy.dot(mo_a*occ_a[:5],mo_a.conj().T)
    #dm_a = reduce(numpy.dot, (mo_a, mo_a.conj().T))
    dm_b = numpy.dot(mo_b, mo_b.conj().T)
    #print(dm_a-numpy.dot(mo_a, mo_a.conj().T))
    return numpy.array((dm_a, dm_b))

class AOCHF(uhf.UHF):
    __doc__ = hf.SCF.__doc__

    def __init__(self, mol, nact=None, nopen=None):
        hf.SCF.__init__(self, mol)
        if nact is not None : self.nact=nact
        if nopen is not None : self.nopen=nopen
        self.nclose=(mol.nelectron-self.nact)//2
        print(self.nact, self.nopen, self.nclose)
        self.nelec=None
    
    get_occ = get_occ
    #@lib.with_doc(make_rdm1.__doc__)
    #def make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
    #    if mo_coeff is None: mo_coeff = self.mo_coeff
    #    if mo_occ is None: mo_occ = self.mo_occ
    #    return make_rdm1(mo_coeff, mo_occ, **kwargs)

