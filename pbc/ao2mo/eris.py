import numpy as np

from pyscf.pbc.dft.gen_grid import gen_uniform_grids
from pyscf.pbc.dft.numint import eval_ao
from pyscf.pbc import tools
from pyscf.lib import logger

def get_ao_pairs_G(cell):
    '''Calculate forward (G|ij) and "inverse" (ij|G) FFT of all AO pairs.

    (G|ij) = \sum_r e^{-iGr} i*(r) j(r)
    (ij|G) = 1/N \sum_r e^{iGr} i(r) j*(r) = 1/N (G|ij).conj()

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
            ao_ij_R = np.einsum('r,r->r', np.conj(aoR[:,i]), aoR[:,j])
            ao_pairs_G[:,ij] = tools.fft(ao_ij_R, cell.gs)
            ao_pairs_invG[:,ij] = tools.ifft(ao_ij_R, cell.gs)
            ij += 1
    return ao_pairs_G, ao_pairs_invG

def get_mo_pairs_G(cell, mo_coeff):
    '''Calculate forward (G|ij) and "inverse" (ij|G) FFT of all MO pairs.

    TODO: - Implement simplifications for real orbitals.
          - Allow for complex orbitals.

    Args:
        mo_coeff: length-2 list of (nao,nmo) ndarrays
            The two sets of MO coefficients to use in calculating the
            product |ij).

    Returns:
        mo_pairs_G, mo_pairs_invG : (ngs, nmoi*nmoj) ndarray
            The FFTs of the real-space MO pairs.
    '''
    coords = gen_uniform_grids(cell)
    aoR = eval_ao(cell, coords) # shape(coords, nao)
    nmoi = mo_coeff[0].shape[1]
    nmoj = mo_coeff[1].shape[1]

    # this also doesn't check for the (common) case
    # where mo_coeff[0] == mo_coeff[1]
    moiR = np.einsum('ri,ia->ra',aoR, mo_coeff[0])
    mojR = np.einsum('ri,ia->ra',aoR, mo_coeff[1])

    # this would need a conj on moiR if we have complex fns
    mo_pairs_R = np.einsum('ri,rj->rij',np.conj(moiR),mojR)
    mo_pairs_G = np.zeros([coords.shape[0],nmoi*nmoj], np.complex128)
    mo_pairs_invG = np.zeros([coords.shape[0],nmoi*nmoj], np.complex128)

    for i in xrange(nmoi):
        for j in xrange(nmoj):
            mo_pairs_G[:,i*nmoj+j] = tools.fft(mo_pairs_R[:,i,j], cell.gs)
            mo_pairs_invG[:,i*nmoj+j] = tools.ifft(mo_pairs_R[:,i,j], cell.gs)
    return mo_pairs_G, mo_pairs_invG

def assemble_eri(cell, orb_pair_G1, orb_pair_invG2, verbose=logger.DEBUG):
    '''Assemble all 4-index electron repulsion integrals.

    (ij|kl) = \sum_G (ij|G) v(G) (G|kl)

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

def get_mo_eri(cell, mo_coeff12, mo_coeff34):
    '''Convenience function to return MO 2-el integrals.'''

    # don't really need FFT and iFFT for both sets
    mo_pairs12_G, mo_pairs12_invG = get_mo_pairs_G(cell, mo_coeff12)
    mo_pairs34_G, mo_pairs34_invG = get_mo_pairs_G(cell, mo_coeff34)
    return assemble_eri(cell, mo_pairs12_G, mo_pairs34_invG)

def get_mo_pairs_G_kpts(cell, mo_coeff_kpts):
    nkpts = mo_coeff_kpts.shape[0]
    ngs = cell.Gv.shape[0]
    nmo = mo_coeff_kpts.shape[2]

    mo_pairs_G_kpts = np.zeros([nkpts, nkpts, ngs, nmo*nmo], np.complex128)
    mo_pairs_invG_kpts = np.zeros([nkpts, nkpts, ngs, nmo*nmo], np.complex128)

    for K in range(nkpts):
        for L in range(nkpts):
            mo_pairs_G_kpts[K,L], mo_pairs_invG_kpts[K,L] = \
            get_mo_pairs_G(cell, [mo_coeff_kpts[K,:,:], mo_coeff_kpts[L,:,:]])

    return mo_pairs_G_kpts, mo_pairs_invG_kpts

def get_mo_eri_kpts(cell, kpts, mo_coeff_kpts):
    '''Assemble *all* MO integrals across kpts'''
    nkpts = mo_coeff_kpts.shape[0]
    nmo = mo_coeff_kpts.shape[2]
    eris=np.zeros([nkpts,nkpts,nkpts,nmo*nmo,nmo*nmo], np.complex128)

    KLMN = tools.get_KLMN(kpts)

    mo_pairs_G_kpts, mo_pairs_invG_kpts = get_mo_pairs_G_kpts(cell, mo_coeff_kpts)

    for K in range(nkpts):
        for L in range(nkpts):
            for M in range(nkpts):
                N = KLMN[K, L, M]
                eris[K,L,M, :, :]=\
                assemble_eri(cell, mo_pairs_G_kpts[K, L],
                             mo_pairs_invG_kpts[M, N])
    return eris
