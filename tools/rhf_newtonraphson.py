#!/usr/bin/env python
#
# Author: Sebastian Wouters <sebastianwouters@gmail.com>
#
# Date: May 18, 2015
#
# Augmented Hessian Newton-Raphson optimization of the RHF energy
#

from pyscf import gto, scf
from pyscf.lib import logger
from pyscf.scf import _vhf

import numpy as np
import scipy
import scipy.sparse.linalg

def __wrapAugmentedHessian( FOCK_mo, numPairs, numVirt ):

    def matvec( vector ):

        xblock  = np.reshape( vector[:-1], ( numPairs, numVirt ), order='F' )
        xscalar = vector[ len(vector)-1 ]
    
        outblock  = 4 * FOCK_mo[:numPairs,numPairs:] * xscalar \
                  + 4 * np.dot( xblock, FOCK_mo[numPairs:,numPairs:] ) \
                  - 4 * np.dot( FOCK_mo[:numPairs,:numPairs], xblock )
    
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
    numPairs = myMF.mol.nelectron / 2
    numVirt  = OEI_ao.shape[0] - numPairs
    numVars  = numPairs * numVirt
    
    if ( dm_guess is None ):
        if (( len( myMF.mo_occ ) == 0 ) or ( safe_guess == True )):
            dm_ao = myMF.get_init_guess( key=myMF.init_guess, mol=myMF.mol )
        else:
            dm_ao = np.dot( np.dot( myMF.mo_coeff, np.diag( myMF.mo_occ ) ), myMF.mo_coeff.T )
    else:
        dm_ao = np.array( dm_guess, copy=True )

    vhf_ao  = myMF.get_veff( myMF.mol, dm_ao )
    FOCK_ao = OEI_ao + vhf_ao
    energies, orbitals = scipy.linalg.eigh( a=FOCK_ao, b=S_ao )
    dm_prev = np.array( dm_ao, copy=True )
    dm_ao   = 2 * np.dot( orbitals[:,:numPairs], orbitals[:,:numPairs].T )
    vhf_ao  = myMF.get_veff( myMF.mol, dm_ao, dm_last=dm_prev, vhf_last=vhf_ao )
    FOCK_ao = OEI_ao + vhf_ao
    FOCK_mo = np.dot( orbitals.T, np.dot( FOCK_ao, orbitals ))
    grdnorm = 4 * np.linalg.norm( FOCK_mo[:numPairs,numPairs:] )
    energy  = myMF.mol.energy_nuc() + 0.5 * np.einsum( 'ij,ij->', OEI_ao + FOCK_ao, dm_ao )
    
    logger.note(myMF, "RHF:NewtonRaphson :: Starting augmented Hessian Newton-Raphson RHF.")
    
    iteration = 0

    while ( grdnorm > 1e-8 ):
    
        iteration += 1
        AugmentedHessian = scipy.sparse.linalg.LinearOperator( ( numVars+1, numVars+1 ), __wrapAugmentedHessian( FOCK_mo, numPairs, numVirt ), dtype=float )
        ini_guess = np.ones( [ numVars+1 ], dtype=float )
        for occ in range( numPairs ):
            for virt in range( numVirt ):
                ini_guess[ occ + numPairs * virt ] = - FOCK_mo[ occ, numPairs + virt ] / max( FOCK_mo[ numPairs + virt, numPairs + virt ] - FOCK_mo[ occ, occ ], 1e-6 )
        eigenval, eigenvec = scipy.sparse.linalg.eigsh( AugmentedHessian, k=1, which='SA', v0=ini_guess, ncv=1024, maxiter=(numVars+1) )
        eigenvec = eigenvec / eigenvec[ numVars ]
        update   = np.reshape( eigenvec[:-1], ( numPairs, numVirt ), order='F' )
        xmat     = np.zeros( [ OEI_ao.shape[0], OEI_ao.shape[0] ], dtype=float )
        xmat[:numPairs,numPairs:] = - update
        xmat[numPairs:,:numPairs] = update.T
        unitary  = scipy.linalg.expm( xmat )
        orbitals = np.dot( orbitals, unitary )
        dm_prev  = np.array( dm_ao, copy=True )
        dm_ao    = 2 * np.dot( orbitals[:,:numPairs], orbitals[:,:numPairs].T )
        vhf_ao   = myMF.get_veff( myMF.mol, dm_ao, dm_last=dm_prev, vhf_last=vhf_ao )
        FOCK_ao  = OEI_ao + vhf_ao
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
    myMF.hf_energy = energy
    myMF.converged = True
    
    return myMF


