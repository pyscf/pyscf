import numpy
from pyscf.dft.numint import _dot_ao_ao, _dot_ao_dm, BLKSIZE
import pyscf.lib
import pyscf.dft
from pyscf.pbc import tools
#from pyscf.pbc.scf import scfint

def eval_ao(cell, coords, kpt=None, isgga=False, relativity=0, bastart=0,
            bascount=None, non0tab=None, verbose=None):
    '''Collocate AO crystal orbitals (opt. gradients) on the real-space grid.

    Args:
        cell : instance of :class:`Cell`

        coords : (nx*ny*nz, 3) ndarray
            The real-space grid point coordinates.

    Kwargs:
        kpt : (3,) ndarray
            The k-point corresponding to the crystal AO.

    Returns:
        aoR : ([4,] nx*ny*nz, nao=cell.nao_nr()) ndarray 
            The value of the AO crystal orbitals on the real-space grid. If
            isgga=True, also contains the value of the orbitals gradient in the
            x, y, and z directions.  It can be either complex or float array,
            depending on the kpt argument.  If kpt is not given (gamma point),
            aoR is a float array.

    See Also:
        pyscf.dft.numint.eval_ao

    '''
    aoR = 0
    # TODO: this is 1j, not -1j; check for band_ovlp convention
    for L in tools.get_lattice_Ls(cell, cell.nimgs):
        if kpt is None:
            aoR += pyscf.dft.numint.eval_ao(cell, coords-L, isgga, relativity,
                                            bastart, bascount,
                                            non0tab, verbose)
        else:
            factor = numpy.exp(1j*numpy.dot(kpt,L))
            aoR += pyscf.dft.numint.eval_ao(cell, coords-L, isgga, relativity,
                                            bastart, bascount,
                                            non0tab, verbose) * factor

    if cell.ke_cutoff is not None:
        ke = 0.5*numpy.einsum('gi,gi->g', cell.Gv, cell.Gv)
        ke_mask = ke < cell.ke_cutoff

        aoG = numpy.zeros_like(aoR)
        for i in range(cell.nao_nr()):
            if isgga:
                for c in range(4):
                    aoG[c][ke_mask, i] = tools.fft(aoR[c][:,i], cell.gs)[ke_mask]
                    aoR[c][:,i] = tools.ifft(aoG[c][:,i], cell.gs)
            else:
                aoG[ke_mask, i] = tools.fft(aoR[:,i], cell.gs)[ke_mask]
                aoR[:,i] = tools.ifft(aoG[:,i], cell.gs)

    return aoR

def eval_rho(mol, ao, dm, non0tab=None, 
             isgga=False, verbose=None):
    '''Collocate the *real* density (opt. gradients) on the real-space grid.

    Args:
        mol : instance of :class:`Mole` or :class:`Cell`

        ao : ([4,] nx*ny*nz, nao=2*cell.nao_nr()) ndarray 
            The value of the AO crystal orbitals on the real-space grid. If
            isgga=True, also contains the value of the gradient in the x, y,
            and z directions.

    Returns:
        rho : ([4,] nx*ny*nz) ndarray
            The value of the density on the real-space grid. If isgga=True,
            also contains the value of the gradient in the x, y, and z
            directions.

    See Also:
        pyscf.dft.numint.eval_rho

    '''

    assert(ao.flags.c_contiguous)
    if isgga:
        ngrids, nao = ao[0].shape
    else:
        ngrids, nao = ao.shape

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.int8)

    # if isgga:
    #     rho = numpy.empty((4,ngrids))
    #     c0 = _dot_ao_dm(mol, ao[0], dm, nao, ngrids, non0tab)
    #     rho[0] = numpy.einsum('pi,pi->p', ao[0], c0)
    #     for i in range(1, 4):
    #         c1 = _dot_ao_dm(mol, ao[i], dm, nao, ngrids, non0tab)
    #         rho[i] = numpy.einsum('pi,pi->p', ao[0], c1) * 2 # *2 for +c.c.
    # else:
    #     c0 = _dot_ao_dm(mol, ao, dm, nao, ngrids, non0tab)
    #     rho = numpy.einsum('pi,pi->p', ao, c0)
    # return rho

    # if isgga:
    #     ngrids, nao = ao[0].shape
    # else:
    #     ngrids, nao = ao.shape

    # if non0tab is None:
    #     print "this PATH"
    #     non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
    #                          dtype=numpy.int8)

    # #if ao[0].dtype==numpy.complex128: # complex orbitals
    # if True:
    #     #dm_re = numpy.ascontiguousarray(dm.real)
    #     dm_re = dm
    #     rho = numpy.empty((4,ngrids))
    #     #ao_re = numpy.ascontiguousarray(ao[0].real)
    #     ao_re = ao[0]
    #     c0_rr = _dot_ao_dm(mol, ao_re, dm_re, nao, ngrids, non0tab)

    #     rho[0] = (numpy.einsum('pi,pi->p', ao_re, c0_rr))

    #     for i in range(1, 4):
    #         c1 = _dot_ao_dm(mol, ao[i], dm, nao, ngrids, non0tab)
    #         rho[i] = numpy.einsum('pi,pi->p', ao[0], c1) * 2 # *2 for +c.c.

    #     for i in range(1, 4):
    #     #     #ao_re = numpy.ascontiguousarray(ao[i].real)
    #         ao_re = ao[i]

    #         c1_rr = _dot_ao_dm(mol, ao_re, dm_re, nao, ngrids, non0tab)

    #         rho[i] = (numpy.einsum('pi,pi->p', ao_re, c1_rr)) *2
    #     return rho

    # complex orbitals or density matrix
    if numpy.iscomplexobj(ao) or numpy.iscomplexobj(dm):

        dm_re = numpy.ascontiguousarray(dm.real)
        dm_im = numpy.ascontiguousarray(dm.imag)

        if isgga:
            rho = numpy.empty((4,ngrids))
            ao0_re = numpy.ascontiguousarray(ao[0].real)
            ao0_im = numpy.ascontiguousarray(ao[0].imag)

            # DM * ket: e.g. ir denotes dm_im | ao_re >
            c0_rr = _dot_ao_dm(mol, ao0_re, dm_re, nao, ngrids, non0tab)
            c0_ri = _dot_ao_dm(mol, ao0_im, dm_re, nao, ngrids, non0tab)
            c0_ir = _dot_ao_dm(mol, ao0_re, dm_im, nao, ngrids, non0tab)
            c0_ii = _dot_ao_dm(mol, ao0_im, dm_im, nao, ngrids, non0tab)

            # bra * DM
            rho[0] = (numpy.einsum('pi,pi->p', ao0_im, c0_ri) +
                      numpy.einsum('pi,pi->p', ao0_re, c0_rr) +
                      numpy.einsum('pi,pi->p', ao0_im, c0_ir) -
                      numpy.einsum('pi,pi->p', ao0_re, c0_ii))

            for i in range(1, 4):
                # ao_re = numpy.ascontiguousarray(ao[i].real)
                # ao_im = numpy.ascontiguousarray(ao[i].imag)
                ao_re = numpy.ascontiguousarray(ao[i].real)
                ao_im = numpy.ascontiguousarray(ao[i].imag)
    
                c1_rr = _dot_ao_dm(mol, ao_re, dm_re, nao, ngrids, non0tab)
                c1_ri = _dot_ao_dm(mol, ao_im, dm_re, nao, ngrids, non0tab)
                c1_ir = _dot_ao_dm(mol, ao_re, dm_im, nao, ngrids, non0tab)
                c1_ii = _dot_ao_dm(mol, ao_im, dm_im, nao, ngrids, non0tab)

                rho[i] = (numpy.einsum('pi,pi->p', ao0_im, c1_ri) +
                          numpy.einsum('pi,pi->p', ao0_re, c1_rr) +
                          numpy.einsum('pi,pi->p', ao0_im, c1_ir) -
                          numpy.einsum('pi,pi->p', ao0_re, c1_ii)) * 2 # *2 for +c.c.       
        else:
            ao_re = numpy.ascontiguousarray(ao.real)
            ao_im = numpy.ascontiguousarray(ao.imag)
            # DM * ket: e.g. ir denotes dm_im | ao_re >
            
            c0_rr = _dot_ao_dm(mol, ao_re, dm_re, nao, ngrids, non0tab)
            c0_ri = _dot_ao_dm(mol, ao_im, dm_re, nao, ngrids, non0tab)
            c0_ir = _dot_ao_dm(mol, ao_re, dm_im, nao, ngrids, non0tab)
            c0_ii = _dot_ao_dm(mol, ao_im, dm_im, nao, ngrids, non0tab)
            # bra * DM
            rho = (numpy.einsum('pi,pi->p', ao_im, c0_ri) +
                   numpy.einsum('pi,pi->p', ao_re, c0_rr) +
                   numpy.einsum('pi,pi->p', ao_im, c0_ir) -
                   numpy.einsum('pi,pi->p', ao_re, c0_ii))

    # real orbitals and real DM
    else:
        rho = pyscf.dft.numint.eval_rho(mol, ao, dm, non0tab, isgga, verbose)

    return rho

def eval_mat(mol, ao, weight, rho, vrho, vsigma=None, non0tab=None,
             isgga=False, verbose=None):
    '''Calculate the XC potential AO matrix.

    Args:
        mol : instance of :class:`Mole` or :class:`Cell`

        ao : ([4,] nx*ny*nz, nao=cell.nao_nr()) ndarray 
            The value of the AO crystal orbitals on the real-space grid. If
            isgga=True, also contains the value of the gradient in the x, y,
            and z directions.

    Returns:
        rho : ([4,] nx*ny*nz) ndarray
            The value of the density on the real-space grid. If isgga=True,
            also contains the value of the gradient in the x, y, and z
            directions.
    
    See Also:
        pyscf.dft.numint.eval_mat

    '''

    if isgga:
        ngrids, nao = ao[0].shape
    else:
        ngrids, nao = ao.shape

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.int8)

    if numpy.iscomplexobj(ao):
        if isgga:
            assert(vsigma is not None and rho.ndim==2)
            #wv = weight * vsigma * 2
            #aow  = numpy.einsum('pi,p->pi', ao[1], rho[1]*wv)
            #aow += numpy.einsum('pi,p->pi', ao[2], rho[2]*wv)
            #aow += numpy.einsum('pi,p->pi', ao[3], rho[3]*wv)
            #aow += numpy.einsum('pi,p->pi', ao[0], .5*weight*vrho)
            wv = numpy.empty_like(rho)
            wv[0]  = weight * vrho * .5
            wv[1:] = rho[1:] * (weight * vsigma * 2)
            aow = numpy.einsum('npi,np->pi', ao, wv)

            ao_re = numpy.ascontiguousarray(ao[0].real)
            ao_im = numpy.ascontiguousarray(ao[0].imag)

            aow_re = numpy.ascontiguousarray(aow.real)
            aow_im = numpy.ascontiguousarray(aow.imag)

        else:
            # *.5 because return mat + mat.T
            #:aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho)
            ao_re = numpy.ascontiguousarray(ao.real)
            ao_im = numpy.ascontiguousarray(ao.imag)

            aow_re = ao_re * (.5*weight*vrho).reshape(-1,1)
            aow_im = ao_im * (.5*weight*vrho).reshape(-1,1)
            #mat = pyscf.lib.dot(ao.T, aow)

        mat_re = _dot_ao_ao(mol, ao_re, aow_re, nao, ngrids, non0tab)
        mat_re += _dot_ao_ao(mol, ao_im, aow_im, nao, ngrids, non0tab)
        mat_im = _dot_ao_ao(mol, ao_re, aow_im, nao, ngrids, non0tab)
        mat_im -= _dot_ao_ao(mol, ao_im, aow_re, nao, ngrids, non0tab)

        mat = mat_re + 1j*mat_im

        # print "MATRIX", mat.dtype
        #return (mat + mat.T.conj()).real
        # print "MAT DTYPE", mat.dtype
        # print "HACK MAT"
        return (mat + mat.T.conj())
        #return 2 * mat
        
    else:
        return pyscf.dft.numint.eval_mat(mol, ao, 
                                         weight, rho, vrho, 
                                         vsigma=None, non0tab=None,
                                         isgga=False, verbose=None)


class _NumInt(pyscf.dft.numint._NumInt):
    '''Generalization of pyscf's _NumInt class for a single k-point shift and
    periodic images.
    '''
    def __init__(self, kpt=None):
        pyscf.dft.numint._NumInt.__init__(self)
        self.kpt = kpt

    def eval_ao(self, mol, coords, isgga=False, relativity=0, bastart=0,
                bascount=None, non0tab=None, verbose=None):
        return eval_ao(mol, coords, self.kpt, isgga, relativity, bastart, 
                       bascount, non0tab, verbose)

    def eval_rho(self, mol, ao, dm, non0tab=None, isgga=False, verbose=None):
        return eval_rho(mol, ao, dm, non0tab, isgga, verbose)

    def eval_rho2(self, mol, ao, dm, non0tab=None, isgga=False, verbose=None):
        raise NotImplementedError

    def nr_rks(self, mol, grids, x_id, c_id, dms, hermi=1,
               max_memory=2000, verbose=None):
        '''
        Use slow function in numint, which only calls eval_rho, eval_mat.
        Faster function uses eval_rho2 which is not yet implemented.
        '''
        return pyscf.dft.numint.nr_rks_vxc(self, mol, grids, x_id, c_id, dms, 
                                           spin=0, relativity=0, hermi=1,
                                           max_memory=max_memory, verbose=verbose)

    def nr_uks(self, mol, grids, x_id, c_id, dms, hermi=1,
               max_memory=2000, verbose=None):
        raise NotImplementedError

    def eval_mat(self, mol, ao, weight, rho, vrho, vsigma=None, non0tab=None,
                 isgga=False, verbose=None):
        # use local function for complex eval_mat
        return eval_mat(mol, ao, weight, rho, vrho, vsigma, non0tab,
                        isgga, verbose)


class _KNumInt(pyscf.dft.numint._NumInt):
    '''Generalization of pyscf's _NumInt class for k-point sampling and 
    periodic images.
    '''
    def __init__(self, kpts=None):
        pyscf.dft.numint._NumInt.__init__(self)
        self.kpts = kpts

    def eval_ao(self, mol, coords, isgga=False, relativity=0, bastart=0,
                bascount=None, non0tab=None, verbose=None):
        '''
        Returns:
            ao_kpts: (nkpts, ngs, nao) ndarray 
                AO values at each k-point
        '''
        nkpts = len(self.kpts)
        ngs = len(coords)
        nao = mol.nao_nr()

        ao_kpts = numpy.empty([nkpts, ngs, nao],numpy.complex128)
        for k in range(nkpts):
            kpt = self.kpts[k,:]
            ao_kpts[k,:,:] = eval_ao(mol, coords, kpt, isgga,
                                  relativity, bastart, bascount,
                                  non0tab, verbose)
        return ao_kpts

    def eval_rho(self, mol, ao_kpts, dm_kpts, non0tab=None,
             isgga=False, verbose=None):
        '''
        Args:
            mol : Mole or Cell object
            ao_kpts : (nkpts, ngs, nao) ndarray
                AO values at each k-point
            dm_kpts: (nkpts, nao, nao) ndarray
                Density matrix at each k-point

        Returns:
           rhoR : (ngs,) ndarray
        '''
        nkpts, ngs, nao = ao_kpts.shape
        rhoR = numpy.zeros(ngs)
        for k in range(nkpts):
            rhoR += 1./nkpts*eval_rho(mol, ao_kpts[k,:,:], dm_kpts[k,:,:])
        return rhoR

    def eval_rho2(self, mol, ao, dm, non0tab=None, isgga=False,
                  verbose=None):
        raise NotImplementedError

    def nr_rks(self, mol, grids, x_id, c_id, dms, hermi=1,
               max_memory=2000, verbose=None):
        '''
        Use slow function in numint, which only calls eval_rho, eval_mat.
        Faster function uses eval_rho2 which is not yet implemented.
        '''
        # TODO: fix spin, relativity
        spin=0; relativity=0
        return pyscf.dft.numint.nr_rks_vxc(self, mol, grids, x_id, c_id, dms,
                                           spin, relativity, hermi,
                                           max_memory, verbose)

    def nr_uks(self, mol, grids, x_id, c_id, dms, hermi=1,
               max_memory=2000, verbose=None):
        raise NotImplementedError

    def eval_mat(self, mol, ao, weight, rho, vrho, vsigma=None, non0tab=None,
                 isgga=False, verbose=None):
        # use local function for complex eval_mat
        nkpts = len(self.kpts)
        nao = ao.shape[2]

        mat = numpy.zeros((nkpts, nao, nao), dtype=ao.dtype)
        for k in range(nkpts):
            mat[k,:,:] = eval_mat(mol, ao[k,:,:], weight,
                                    rho, vrho, vsigma, non0tab,
                                    isgga, verbose)
        return mat

