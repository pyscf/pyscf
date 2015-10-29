import numpy as np

from pyscf.pbc.dft.gen_grid import gen_uniform_grids
from pyscf.pbc.dft.numint import eval_ao
from pyscf.pbc import tools
from pyscf.lib import logger

def get_ao_pairs_G(cell):
    '''Calculate forward (G|ij) and "inverse" (ij|G) FFT of all AO pairs.

    (G|ij) = \sum_r e^{-iGr} i(r) j(r)
    (ij|G) = 1/N \sum_r e^{iGr} i*(r) j*(r) = 1/N (G|ij).conj()

    Args:
        cell : instance of :class:`Cell`

    Returns:
        ao_pairs_G, ao_pairs_invG : (ngs, nao*(nao+1)/2) ndarray
            The FFTs of the real-space AO pairs.

    '''
    coords = gen_uniform_grids(cell)
    aoR = eval_ao(cell, coords) # shape = (coords, nao)
    nao = aoR.shape[1]
    npair = nao*(nao+1)/2
    ao_pairs_G = np.zeros([coords.shape[0], npair], np.complex128)
    ao_pairs_invG = np.zeros([coords.shape[0], npair], np.complex128)
    ij = 0
    for i in range(nao):
        for j in range(i+1):
            ao_ij_R = np.einsum('r,r->r', aoR[:,i], aoR[:,j])
            ao_pairs_G[:,ij] = tools.fft(ao_ij_R, cell.gs)         
            ao_pairs_invG[:,ij] = tools.ifft(ao_ij_R, cell.gs)
            ij += 1
    return ao_pairs_G, ao_pairs_invG
    
def get_mo_pairs_G(cell, mo_coeffs):
    '''Calculate forward (G|ij) and "inverse" (ij|G) FFT of all MO pairs.
    
    TODO: - Implement simplifications for real orbitals.
          - Allow for complex orbitals.

    Args:
        mo_coeffs: length-2 list of (nao,nmo) ndarrays
            The two sets of MO coefficients to use in calculating the 
            product |ij).

    Returns:
        mo_pairs_G, mo_pairs_invG : (ngs, nmoi*nmoj) ndarray
            The FFTs of the real-space MO pairs.
    '''
    coords = gen_uniform_grids(cell)
    aoR = eval_ao(cell, coords) # shape(coords, nao)
    nmoi = mo_coeffs[0].shape[1]
    nmoj = mo_coeffs[1].shape[1]

    # this also doesn't check for the (common) case
    # where mo_coeffs[0] == mo_coeffs[1]
    moiR = np.einsum('ri,ia->ra',aoR, mo_coeffs[0])
    mojR = np.einsum('ri,ia->ra',aoR, mo_coeffs[1])

    # this would need a conj on moiR if we have complex fns
    mo_pairs_R = np.einsum('ri,rj->rij',moiR,mojR)
    mo_pairs_G = np.zeros([coords.shape[0],nmoi*nmoj], np.complex128)
    mo_pairs_invG = np.zeros([coords.shape[0],nmoi*nmoj], np.complex128)

    for i in xrange(nmoi):
        for j in xrange(nmoj):
            mo_pairs_G[:,i*nmoj+j] = tools.fft(mo_pairs_R[:,i,j], cell.gs)
            mo_pairs_invG[:,i*nmoj+j] = tools.ifft(mo_pairs_R[:,i,j], cell.gs)
    return mo_pairs_G, mo_pairs_invG

def assemble_eri(cell, orb_pair_G1, orb_pair_invG2, verbose=logger.DEBUG):
    '''Assemble all 4-index electron repulsion integrals.

    (ij|kl) = \sum_G (ij|G)(G|kl) 

    Returns:
        (nmo1*nmo2, nmo3*nmo4) ndarray

    '''
    log = logger.Logger
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(cell.stdout, verbose)

    log.debug('Performing periodic ERI assembly of (%i, %i) ij pairs', 
              orb_pair_G1.shape[1], orb_pair_invG2.shape[1])
    coulG = tools.get_coulG(cell)
    ngs = orb_pair_invG2.shape[0]
    Jorb_pair_invG2 = np.einsum('g,gn->gn',coulG,orb_pair_invG2)*(cell.vol/ngs)
    eri = np.einsum('gm,gn->mn',orb_pair_G1, Jorb_pair_invG2)
    return eri

def get_ao_eri(cell):
    '''Convenience function to return AO 2-el integrals.'''

    ao_pairs_G, ao_pairs_invG = get_ao_pairs_G(cell)
    return assemble_eri(cell, ao_pairs_G, ao_pairs_invG)
        
def get_mo_eri(cell, mo_coeffs12, mo_coeffs34):
    '''Convenience function to return MO 2-el integrals.'''

    # don't really need FFT and iFFT for both sets
    mo_pairs12_G, mo_pairs12_invG = get_mo_pairs_G(cell, mo_coeffs12)
    mo_pairs34_G, mo_pairs34_invG = get_mo_pairs_G(cell, mo_coeffs34)
    return assemble_eri(cell, mo_pairs12_G, mo_pairs34_invG)

