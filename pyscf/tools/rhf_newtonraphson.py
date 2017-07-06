#!/usr/bin/env python
#
# Author: Sebastian Wouters <sebastianwouters@gmail.com>
#
# Date: May 18, 2015
#
# Augmented Hessian Newton-Raphson optimization of the RHF energy
#
# The gradient and hessian were determined from the equations in
# http://sebwouters.github.io/CheMPS2/doxygen/classCheMPS2_1_1CASSCF.html
# by throwing out all active space components.
#
# In the following:
#    * (o,p) denote occupied spatial RHF orbitals
#    * (v,w) denote virtual  spatial RHF orbitals
#    * f is the Fock operator
#
# \frac{\partial E}{\partial x_{ov}} = 4 * f_{vo}
#
# \frac{\partial^2 E}{\partial x_{ov} \partial x_{pw}} = 4 * delta_{op} * f_{vw}
#                                                      - 4 * delta_{vw} * f_{op}
#                                                      + 4 * [ 4 * (vo|wp) - (vw|op) - (vp|wo) ]
#

from pyscf import gto, scf
from pyscf.lib import logger
from pyscf.lib import linalg_helper
from pyscf.scf import _vhf

import numpy as np
import scipy
import scipy.sparse.linalg

class __JKengine:

    def __init__( self, myMF, orbitals=None ):
    
        self.mf = myMF
        self.dm_prev  = 0
        self.vhf_prev = 0
        self.orbs     = orbitals
        #self.iter     = 0
        
    def getJK_mo( self, dm_mo ):
    
        dm_ao = np.dot( np.dot( self.orbs, dm_mo ), self.orbs.T )
        JK_ao = self.mf.get_veff( self.mf.mol, dm_ao, dm_last=self.dm_prev, vhf_last=self.vhf_prev )
        self.dm_prev  = dm_ao
        self.vhf_prev = JK_ao
        JK_mo = np.dot( np.dot( self.orbs.T, JK_ao ), self.orbs )
        #self.iter += 1
        return JK_mo
        
    def getJK_ao( self, dm_ao ):
    
        JK_ao = self.mf.get_veff( self.mf.mol, dm_ao, dm_last=self.dm_prev, vhf_last=self.vhf_prev )
        self.dm_prev  = dm_ao
        self.vhf_prev = JK_ao
        #self.iter += 1
        return JK_ao

def __wrapAugmentedHessian( FOCK_mo, numPairs, numVirt, myJK_mo ):

    def matvec( vector ):

        xblock  = np.reshape( vector[:-1], ( numPairs, numVirt ), order='F' )
        xscalar = vector[ len(vector)-1 ]
    
        outblock  = 4 * FOCK_mo[:numPairs,numPairs:] * xscalar \
                  + 4 * np.dot( xblock, FOCK_mo[numPairs:,numPairs:] ) \
                  - 4 * np.dot( FOCK_mo[:numPairs,:numPairs], xblock )
        fakedens  = np.zeros( [ FOCK_mo.shape[0], FOCK_mo.shape[0] ], dtype=float )
        fakedens[:numPairs,numPairs:] = xblock
        fakedens[numPairs:,:numPairs] = xblock.T
        outblock += 8 * ( myJK_mo.getJK_mo( fakedens )[:numPairs,numPairs:] )
    
        result = np.zeros( [ len(vector) ], dtype=float )
        result[ len(vector)-1 ] = 4 * np.einsum( 'ij,ij->', xblock, FOCK_mo[:numPairs,numPairs:] )
        result[ :-1 ] = np.reshape( outblock, ( numPairs * numVirt ), order='F' )
    
        return result
        
    return matvec

def solve( myMF, dm_guess=None, safe_guess=True ):

    assert(hasattr(myMF, 'mol'))
    assert(hasattr(myMF, 'mo_occ'))
    assert(hasattr(myMF, 'mo_coeff'))
    assert(myMF.mol.nelectron % 2 == 0) # RHF possible
    
    S_ao     = myMF.get_ovlp( myMF.mol )
    OEI_ao   = myMF.get_hcore( myMF.mol )
    numPairs = myMF.mol.nelectron // 2
    numVirt  = OEI_ao.shape[0] - numPairs
    numVars  = numPairs * numVirt
    
    if ( dm_guess is None ):
        if (( len( myMF.mo_occ ) == 0 ) or ( safe_guess == True )):
            dm_ao = myMF.get_init_guess( key=myMF.init_guess, mol=myMF.mol )
        else:
            dm_ao = np.dot( np.dot( myMF.mo_coeff, np.diag( myMF.mo_occ ) ), myMF.mo_coeff.T )
    else:
        dm_ao = np.array( dm_guess, copy=True )
        
    myJK_ao = __JKengine( myMF )
    FOCK_ao = OEI_ao + myJK_ao.getJK_ao( dm_ao )
    energies, orbitals = scipy.linalg.eigh( a=FOCK_ao, b=S_ao )
    dm_ao   = 2 * np.dot( orbitals[:,:numPairs], orbitals[:,:numPairs].T )
    FOCK_ao = OEI_ao + myJK_ao.getJK_ao( dm_ao )
    FOCK_mo = np.dot( orbitals.T, np.dot( FOCK_ao, orbitals ))
    grdnorm = 4 * np.linalg.norm( FOCK_mo[:numPairs,numPairs:] )
    energy  = myMF.mol.energy_nuc() + 0.5 * np.einsum( 'ij,ij->', OEI_ao + FOCK_ao, dm_ao )
    
    logger.note(myMF, "RHF:NewtonRaphson :: Starting augmented Hessian Newton-Raphson RHF.")
    
    iteration = 0

    while ( grdnorm > 1e-7 ):
    
        iteration += 1
        tempJK_mo = __JKengine( myMF, orbitals )
        ini_guess = np.ones( [ numVars+1 ], dtype=float )
        for occ in range( numPairs ):
            for virt in range( numVirt ):
                ini_guess[ occ + numPairs * virt ] = - FOCK_mo[ occ, numPairs + virt ] / max( FOCK_mo[ numPairs + virt, numPairs + virt ] - FOCK_mo[ occ, occ ], 1e-6 )
        
        def myprecon( resid, eigval, eigvec ):
            
            myprecon_cutoff = 1e-10
            local_myprecon = np.zeros( [ numVars+1 ], dtype=float )
            for occ in range( numPairs ):
                for virt in range( numVirt ):
                    denominator = FOCK_mo[ numPairs + virt, numPairs + virt ] - FOCK_mo[ occ, occ ] - eigval
                    if ( abs( denominator ) < myprecon_cutoff ):
                        local_myprecon[ occ + numPairs * virt ] = eigvec[ occ + numPairs * virt ] / myprecon_cutoff
                    else:
                        # local_myprecon = eigvec / ( diag(H) - eigval ) = K^{-1} u
                        local_myprecon[ occ + numPairs * virt ] = eigvec[ occ + numPairs * virt ] / denominator
            if ( abs( eigval ) < myprecon_cutoff ):
                local_myprecon[ numVars ] = eigvec[ numVars ] / myprecon_cutoff
            else:
                local_myprecon[ numVars ] = - eigvec[ numVars ] / eigval
            # alpha_myprecon = - ( r, K^{-1} u ) / ( u, K^{-1} u )
            alpha_myprecon = - np.einsum( 'i,i->', local_myprecon, resid ) / np.einsum( 'i,i->', local_myprecon, eigvec )
            # local_myprecon = r - ( r, K^{-1} u ) / ( u, K^{-1} u ) * u
            local_myprecon = resid + alpha_myprecon * eigvec
            for occ in range( numPairs ):
                for virt in range( numVirt ):
                    denominator = FOCK_mo[ numPairs + virt, numPairs + virt ] - FOCK_mo[ occ, occ ] - eigval
                    if ( abs( denominator ) < myprecon_cutoff ):
                        local_myprecon[ occ + numPairs * virt ] = - local_myprecon[ occ + numPairs * virt ] / myprecon_cutoff
                    else:
                        local_myprecon[ occ + numPairs * virt ] = - local_myprecon[ occ + numPairs * virt ] / denominator
            if ( abs( eigval ) < myprecon_cutoff ):
                local_myprecon[ numVars ] = - local_myprecon[ occ + numPairs * virt ] / myprecon_cutoff
            else:
                local_myprecon[ numVars ] = local_myprecon[ occ + numPairs * virt ] / eigval
            return local_myprecon
        
        eigenval, eigenvec = linalg_helper.davidson( aop=__wrapAugmentedHessian( FOCK_mo, numPairs, numVirt, tempJK_mo ), \
                                                     x0=ini_guess, \
                                                     precond=myprecon, \
                                                     #tol=1e-14, \
                                                     #max_cycle=50, \
                                                     max_space=20, \
                                                     #lindep=1e-16, \
                                                     #max_memory=2000, \
                                                     nroots=1 )
        
        #logger.note(myMF, "   RHF:NewtonRaphson :: # JK computs  (iteration %d) = %d", iteration, tempJK_mo.iter)
        eigenvec = eigenvec / eigenvec[ numVars ]
        update   = np.reshape( eigenvec[:-1], ( numPairs, numVirt ), order='F' )
        xmat     = np.zeros( [ OEI_ao.shape[0], OEI_ao.shape[0] ], dtype=float )
        xmat[:numPairs,numPairs:] = - update
        xmat[numPairs:,:numPairs] = update.T
        unitary  = scipy.linalg.expm( xmat )
        orbitals = np.dot( orbitals, unitary )
        dm_ao    = 2 * np.dot( orbitals[:,:numPairs], orbitals[:,:numPairs].T )
        FOCK_ao  = OEI_ao + myJK_ao.getJK_ao( dm_ao )
        FOCK_mo  = np.dot( orbitals.T, np.dot( FOCK_ao, orbitals ))
        grdnorm  = 4 * np.linalg.norm( FOCK_mo[:numPairs,numPairs:] )
        energy   = myMF.mol.energy_nuc() + 0.5 * np.einsum( 'ij,ij->', OEI_ao + FOCK_ao, dm_ao )
        logger.note(myMF, "   RHF:NewtonRaphson :: gradient norm (iteration %d) = %1.3g" , iteration, grdnorm)
        logger.note(myMF, "   RHF:NewtonRaphson :: RHF energy    (iteration %d) = %1.15g", iteration, energy)
    
    logger.note(myMF, "RHF:NewtonRaphson :: Convergence reached.")
    logger.note(myMF, "RHF:NewtonRaphson :: Converged RHF energy = %1.15g", energy)
    
    energies, orbitals = scipy.linalg.eigh( a=FOCK_ao, b=S_ao )
    
    myMF.mo_coeff  = orbitals
    myMF.mo_occ    = np.zeros( [ OEI_ao.shape[0] ], dtype=int )
    myMF.mo_occ[:numPairs] = 2
    myMF.mo_energy = energies
    myMF.e_tot = energy
    myMF.converged = True
    
    return myMF


