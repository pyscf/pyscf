import pyscf.dft
from pyscf.lib import logger

import numpy as np
import scipy.linalg
import scipy.special
import scipy.optimize

def get_Gv(cell):
    '''Calculate three-dimensional G-vectors for a given cell; see MH (3.8).

    Indices along each direction go as [0...cell.gs, -cell.gs...-1]
    to follow FFT convention. Note that, for each direction, ngs = 2*cell.gs+1.

    Args:
        cell : instance of :class:`Cell`

    Returns:
        Gv : (3, ngs) ndarray of floats
            The array of G-vectors.

    '''
    invhT = scipy.linalg.inv(cell.h.T)

    gxrange = range(cell.gs[0]+1)+range(-cell.gs[0],0)
    gyrange = range(cell.gs[1]+1)+range(-cell.gs[1],0)
    gzrange = range(cell.gs[2]+1)+range(-cell.gs[2],0)
    gxyz = _span3(gxrange, gyrange, gzrange)

    Gv = 2*np.pi*np.dot(invhT,gxyz)
    return Gv

def get_SI(cell, Gv):
    '''Calculate the structure factor for all atoms; see MH (3.34).

    Args:
        cell : instance of :class:`Cell`

        Gv : (3, ngs) ndarray of floats
            The array of G-vectors.

    Returns:
        SI : (natm, ngs) ndarray, dtype=np.complex128
            The structure factor for each atom at each G-vector.

    '''
    ngs = Gv.shape[1]
    SI = np.empty([cell.natm, ngs], np.complex128)
    for ia in range(cell.natm):
        SI[ia,:] = np.exp(-1j*np.dot(Gv.T, cell.atom_coord(ia)))
    return SI

def get_coulG(cell):
    '''Calculate the Coulomb kernel 4*pi/G^2 for all G-vectors (0 for G=0).

    Args:
        cell : instance of :class:`Cell`

    Returns:
        coulG : (ngs,) ndarray
            The Coulomb kernel.

    '''
    Gv = get_Gv(cell)
    absG2 = np.einsum('ij,ij->j',np.conj(Gv),Gv)
    with np.errstate(divide='ignore'):
        coulG = 4*np.pi/absG2
    coulG[0] = 0.

    return coulG

def _span3(*xs):
    '''Generate integer coordinates for each three-dimensional grid point.

    Args:
        *xs : length-3 tuple of np.arange() arrays
            The integer coordinates along each direction.

    Returns:
         (3, ngx*ngy*ngz) ndarray
            The integer coordinates for each grid point.

    Examples:

    >>> _span3(np.array([2,3,2]))
    array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2],
           [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])

    '''
    c = np.empty([3]+[len(x) for x in xs])
    c[0,:,:,:] = np.asarray(xs[0]).reshape(-1,1,1)
    c[1,:,:,:] = np.asarray(xs[1]).reshape(1,-1,1)
    c[2,:,:,:] = np.asarray(xs[2]).reshape(1,1,-1)
    return c.reshape(3,-1)

def fft(f, gs):
    '''Perform the 3D FFT from real (R) to reciprocal (G) space.

    Re: MH (3.25), we assume Ns := ngs = 2*gs+1

    After FFT, (u, v, w) -> (j, k, l).
    (jkl) is in the index order of Gv.

    FFT normalization factor is 1., as in MH and in `numpy.fft`.

    Args:
        f : (nx*ny*nz,) ndarray
            The function to be FFT'd, flattened to a 1D array corresponding
            to the index order of `_span3`.
        gs : (3,) ndarray of ints
            The number of *positive* G-vectors along each direction.
    
    Returns:
        (nx*ny*nz,) ndarray
            The FFT 1D array in same index order as Gv (natural order of
            numpy.fft).

    '''
    ngs = 2*gs+1
    f3d = np.reshape(f, ngs)
    g3d = np.fft.fftn(f3d)
    return np.ravel(g3d)

def ifft(g, gs):
    '''Perform the 3D inverse FFT from reciprocal (G) space to real (R) space.

    Inverse FFT normalization factor is 1./N, same as in `numpy.fft` but
    **different** from MH (they use 1.).

    Args:
        g : (nx*ny*nz,) ndarray
            The function to be inverse FFT'd, flattened to a 1D array
            corresponding to the index order of `_span3`.
        gs : (3,) ndarray of ints
            The number of *positive* G-vectors along each direction.
    
    Returns:
        (nx*ny*nz,) ndarray
            The inverse FFT 1D array in same index order as Gv (natural order
            of numpy.fft).

    '''
    ngs = 2*gs+1
    g3d = np.reshape(g, ngs)
    f3d = np.fft.ifftn(g3d)
    return np.ravel(f3d)
    
def ewald(cell, ew_eta, ew_cut, verbose=logger.DEBUG):
    '''Perform real (R) and reciprocal (G) space Ewald sum for the energy.

    Formulation of Martin, App. F2.

    Args:
        cell : instance of :class:`Cell`

        ew_eta, ew_cut : float
            The Ewald 'eta' and 'cut' parameters.

    Returns:
        float
            The Ewald energy consisting of overlap, self, and G-space sum.

    See Also:
        ewald_params
        
    '''
    log = logger.Logger
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(cell.stdout, verbose)

    chargs = [cell.atom_charge(i) for i in range(len(cell._atm))]
    coords = [cell.atom_coord(i) for i in range(len(cell._atm))]

    ewovrl = 0.

    # set up real-space lattice indices [-ewcut ... ewcut]
    ewxrange = range(-ew_cut[0],ew_cut[0]+1)
    ewyrange = range(-ew_cut[1],ew_cut[1]+1)
    ewzrange = range(-ew_cut[2],ew_cut[2]+1)
    ewxyz = _span3(ewxrange,ewyrange,ewzrange)

    # SLOW = True
    # if SLOW == True:
    #     ewxyz = ewxyz.T
    #     for ic, (ix, iy, iz) in enumerate(ewxyz):
    #         L = np.einsum('ij,j->i', cell.h, ewxyz[ic])

    #         # prime in summation to avoid self-interaction in unit cell
    #         if (ix == 0 and iy == 0 and iz == 0):
    #             print "L is", L
    #             for ia in range(cell.natm):
    #                 qi = chargs[ia]
    #                 ri = coords[ia]
    #                 #for ja in range(ia):
    #                 for ja in range(cell.natm):
    #                     if ja != ia:
    #                         qj = chargs[ja]
    #                         rj = coords[ja]
    #                         r = np.linalg.norm(ri-rj)
    #                         ewovrl += qi * qj / r * scipy.special.erfc(ew_eta * r)
    #         else:
    #             for ia in range(cell.natm):
    #                 qi = chargs[ia]
    #                 ri = coords[ia]
    #                 for ja in range(cell.natm):
    #                     qj=chargs[ja]
    #                     rj=coords[ja]
    #                     r=np.linalg.norm(ri-rj+L)
    #                     ewovrl += qi * qj / r * scipy.special.erfc(ew_eta * r)

    # # else:
    nx = len(ewxrange)
    ny = len(ewyrange)
    nz = len(ewzrange)
    Lall = np.einsum('ij,jk->ik', cell.h, ewxyz).reshape(3,nx,ny,nz)
    #exclude the point where Lall == 0
    Lall[:,ew_cut[0],ew_cut[1],ew_cut[2]] = 1e200
    Lall = Lall.reshape(3,nx*ny*nz)
    Lall = Lall.T

    for ia in range(cell.natm):
        qi = chargs[ia]
        ri = coords[ia]
        for ja in range(ia):
            qj = chargs[ja]
            rj = coords[ja]
            r = np.linalg.norm(ri-rj)
            ewovrl += 2 * qi * qj / r * scipy.special.erfc(ew_eta * r)

    for ia in range(cell.natm):
        qi = chargs[ia]
        ri = coords[ia]
        for ja in range(cell.natm):
            qj = chargs[ja]
            rj = coords[ja]
            r1 = ri-rj + Lall
            r = np.sqrt(np.einsum('ji,ji->j', r1, r1))
            ewovrl += (qi * qj / r * scipy.special.erfc(ew_eta * r)).sum()

    ewovrl *= 0.5

    # last line of Eq. (F.5) in Martin 
    ewself  = -1./2. * np.dot(chargs,chargs) * 2 * ew_eta / np.sqrt(np.pi)
    ewself += -1./2. * np.sum(chargs)**2 * np.pi/(ew_eta**2 * cell.vol)
    
    # g-space sum (using g grid) (Eq. (F.6) in Martin, but note errors as below)
    Gv = get_Gv(cell)
    SI = get_SI(cell, Gv)
    ZSI = np.einsum("i,ij->j", chargs, SI)

    # Eq. (F.6) in Martin is off by a factor of 2, the
    # exponent is wrong (8->4) and the square is in the wrong place
    #
    # Formula should be
    #   1/2 * 4\pi / Omega \sum_I \sum_{G\neq 0} |ZS_I(G)|^2 \exp[-|G|^2/4\eta^2]
    # where
    #   ZS_I(G) = \sum_a Z_a exp (i G.R_a)
    # See also Eq. (32) of ewald.pdf at 
    #   http://www.fisica.uniud.it/~giannozz/public/ewald.pdf

    coulG = get_coulG(cell)
    absG2 = np.einsum('ij,ij->j',np.conj(Gv),Gv)

    ZSIG2 = np.abs(ZSI)**2
    expG2 = np.exp(-absG2/(4*ew_eta**2))
    JexpG2 = coulG*expG2
    ewgI = np.dot(ZSIG2,JexpG2)
    ewg = .5*np.sum(ewgI)
    ewg /= cell.vol

    log.debug('Ewald components = %.15g, %.15g, %.15g', ewovrl, ewself, ewg)
    return ewovrl + ewself + ewg

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
    coords = setup_uniform_grids(cell)
    aoR = get_aoR(cell, coords) # shape = (coords, nao)
    nao = aoR.shape[1]
    npair = nao*(nao+1)/2
    ao_pairs_G = np.zeros([coords.shape[0], npair], np.complex128)
    ao_pairs_invG = np.zeros([coords.shape[0], npair], np.complex128)
    ij = 0
    for i in range(nao):
        for j in range(i+1):
            ao_ij_R = np.einsum('r,r->r', aoR[:,i], aoR[:,j])
            ao_pairs_G[:,ij] = fft(ao_ij_R, cell.gs)         
            ao_pairs_invG[:,ij] = ifft(ao_ij_R, cell.gs)
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
    coords = setup_uniform_grids(cell)
    aoR = get_aoR(cell, coords) # shape(coords, nao)
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
            mo_pairs_G[:,i*nmoj+j] = fft(mo_pairs_R[:,i,j], cell.gs)
            mo_pairs_invG[:,i*nmoj+j] = ifft(mo_pairs_R[:,i,j], cell.gs)
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
    coulG = get_coulG(cell)
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

