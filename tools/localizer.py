#!/usr/bin/env python
#
# Author: Sebastian Wouters <sebastianwouters@gmail.com>
#
# Date: March 5, 2015
#
# Augmented Hessian Newton-Raphson optimization of
#    1. either the Edmiston-Ruedenberg localization cost function
#    2. or the Boys localization cost function
# in both cases with an analytic gradient and hessian
#
# Reference: C. Edmiston and K. Ruedenberg, Reviews of Modern Physics 35, 457-464 (1963). http://dx.doi.org/10.1103/RevModPhys.35.457
#            http://sebwouters.github.io/CheMPS2/doxygen/classCheMPS2_1_1EdmistonRuedenberg.html
#

from pyscf import gto, scf
from pyscf.tools import molden
from pyscf.lib import parameters as param
from pyscf.lib import logger
from pyscf.lib import linalg_helper
from pyscf.scf import _vhf
from pyscf import ao2mo

import numpy as np
import scipy
#import time

import ctypes
import _ctypes
from pyscf.lib import load_library
liblocalizer = load_library('liblocalizer')

class localizer:

    def __init__( self, mol, orbital_coeff, thetype, use_full_hessian=True ):
        r'''Initializer for the localization procedure

        Args:
            mol : A molecule which has been built
            orbital_coeff: Set of orthonormal orbitals, expressed in terms of the AO, which should be localized
            thetype: Which cost function to optimize: 'boys' or 'edmiston'
            use_full_hessian: Whether to do augmented Hessian Newton-Raphson (True) or just -gradient/diag(hessian) (False)
        '''

        assert( ( thetype == 'boys' ) or ( thetype == 'edmiston' ) )

        self.themol   = mol
        self.coeff    = orbital_coeff
        self.Norbs    = orbital_coeff.shape[1]
        self.numVars  = ( self.Norbs * ( self.Norbs - 1 ) ) // 2
        self.u        = np.eye( self.Norbs, dtype=float )
        self.verbose  = mol.verbose
        self.use_hess = use_full_hessian
        self.stdout   = mol.stdout
        
        self.gradient = None
        self.grd_norm = 1.0
        #self.ahnr_cnt = 0
        
        self.__which = thetype
        
        if ( self.__which == 'boys' ):
            rvec        = self.themol.intor('cint1e_r_sph', 3)
            self.x_orig = np.dot( np.dot( self.coeff.T, rvec[0] ) , self.coeff )
            self.y_orig = np.dot( np.dot( self.coeff.T, rvec[1] ) , self.coeff )
            self.z_orig = np.dot( np.dot( self.coeff.T, rvec[2] ) , self.coeff )
            self.x_symm = self.x_orig + self.x_orig.T
            self.y_symm = self.x_orig + self.x_orig.T
            self.z_symm = self.x_orig + self.x_orig.T
        
        if ( self.__which == 'edmiston' ):
            self.eri_orig = ao2mo.incore.full( _vhf.int2e_sph( mol._atm, mol._bas, mol._env ), self.coeff )
            self.eri_rot  = None


    def dump_molden( self, filename, orbital_coeff ):
        r'''Create a molden file to inspect a set of orbitals

        Args:
            filename : Filename of the molden file
            orbital_coeff: Set of orthonormal orbitals, expressed in terms of the AO, for which a molden file should be created
        '''

        with open( filename, 'w' ) as thefile:
            molden.header( self.themol, thefile )
            molden.orbital_coeff( self.themol, thefile, orbital_coeff )


    def __update_unitary( self, flatx ):

        squarex = np.zeros( [ self.Norbs, self.Norbs ], dtype=float )
        increment = 0
        for row in range( self.Norbs ):
            for col in range( row+1, self.Norbs ):
                squarex[ row, col ] =   flatx[ increment ]
                squarex[ col, row ] = - flatx[ increment ]
                increment += 1
        additional_unitary = scipy.linalg.expm( squarex )
        self.u = np.dot( self.u, additional_unitary )
        
        if ( self.__which == 'boys' ):
            x_curr = np.dot( np.dot( self.u.T, self.x_orig ), self.u )
            y_curr = np.dot( np.dot( self.u.T, self.y_orig ), self.u )
            z_curr = np.dot( np.dot( self.u.T, self.z_orig ), self.u )
            self.x_symm = np.array( x_curr + x_curr.T, dtype=ctypes.c_double )
            self.y_symm = np.array( y_curr + y_curr.T, dtype=ctypes.c_double )
            self.z_symm = np.array( z_curr + z_curr.T, dtype=ctypes.c_double )
        if ( self.__which == 'edmiston' ):
            self.eri_rot = ao2mo.incore.full( self.eri_orig, self.u, compact=False ).reshape(self.Norbs, self.Norbs, self.Norbs, self.Norbs)


    def __reorthogonalize( self ):
    
        B = self.u + self.u.T
        eigs, vecs = np.linalg.eigh( B )
        C = np.dot(vecs.T, np.dot(self.u, vecs))
        D = np.zeros( C.shape, dtype=float )
        for row in range( C.shape[0] // 2 ):
            cosine = 0.5 * ( C[ 2*row, 2*row ] + C[ 2*row+1, 2*row+1 ] )
            sine   = 0.5 * ( C[ 2*row, 2*row+1 ] - C[ 2*row+1, 2*row ] )
            theta  = np.arctan2( sine, cosine )
            D[ 2*row, 2*row+1 ] = theta
            D[ 2*row+1, 2*row ] = -theta
        F = np.dot(vecs, np.dot(D, vecs.T))
        newU = scipy.linalg.expm( F )
        new_old_diff = np.linalg.norm( newU - self.u )
        logger.debug(self, "Localizer :: Reorthogonalize : 2-norm( exp(log(U)) - U ) = %g", new_old_diff)
        assert( np.linalg.norm( new_old_diff ) < 1e-5 )
        self.u = np.array( newU, copy=True )
        

    def __costfunction( self ):

        value = 0.0
        if ( self.__which == 'boys' ):
            for i in range( self.Norbs ):
                for j in range( i+1, self.Norbs ): # j > i
                    temp = self.x_symm[i,i] - self.x_symm[j,j]
                    value += temp * temp
                    temp = self.y_symm[i,i] - self.y_symm[j,j]
                    value += temp * temp
                    temp = self.z_symm[i,i] - self.z_symm[j,j]
                    value += temp * temp
            value *= -0.25
        if ( self.__which == 'edmiston' ):
            for i in range( self.Norbs ):
                value += self.eri_rot[i,i,i,i]
            value *= -1
        return value


    def __set_gradient( self ):

        self.gradient = np.zeros( [ self.numVars ], dtype=float )
        if ( self.__which == 'boys' ):
            increment = 0
            for p in range( self.Norbs ):
                for q in range( p+1, self.Norbs ):
                    self.gradient[ increment ] = self.x_symm[p,q] * ( self.x_symm[q,q] - self.x_symm[p,p] ) \
                                               + self.y_symm[p,q] * ( self.y_symm[q,q] - self.y_symm[p,p] ) \
                                               + self.z_symm[p,q] * ( self.z_symm[q,q] - self.z_symm[p,p] )
                    increment += 1
            self.gradient *= -self.Norbs
        if ( self.__which == 'edmiston' ):
            increment = 0
            for p in range( self.Norbs ):
                for q in range( p+1, self.Norbs ):
                    self.gradient[ increment ] = 4*(self.eri_rot[q,q,q,p] - self.eri_rot[p,p,p,q])
                    increment += 1
            self.gradient *= -1
        self.grd_norm = np.linalg.norm( self.gradient )


    def __debug_gradient( self ):

        self.__set_gradient()
        original_umatrix = np.array( self.u, copy=True )

        stepsize = 1e-8
        CF_ref = self.__costfunction()
        gradient_numerical = np.zeros( [ self.numVars ], dtype=float )
        for counter in range( self.numVars ):
            self.u = np.array( original_umatrix, copy=True )
            flatx = np.zeros( [ self.numVars ], dtype=float )
            flatx[counter] = stepsize
            self.__update_unitary( flatx )
            CF_step = self.__costfunction()
            gradient_numerical[ counter ] = ( CF_step - CF_ref ) / stepsize

        self.u = np.array( original_umatrix, copy=True )
        flatx = np.zeros( [ self.numVars ], dtype=float )
        self.__update_unitary( flatx )

        logger.debug(self, "2-norm( gradient difference ) = %g", np.linalg.norm( self.gradient - gradient_numerical ))
        logger.debug(self, "2-norm( gradient )            = %g", np.linalg.norm( self.gradient ))
        
    def __diag_hessian( self ):
    
        diagonal = np.zeros( [ self.numVars ], dtype=float )
        if ( self.__which == 'boys' ):
            increment = 0
            for p in range( self.Norbs ):
                for q in range( p+1, self.Norbs ):
                    diagonal[ increment ] = 4 * self.x_symm[p,q] * self.x_symm[p,q] \
                                          + 4 * self.y_symm[p,q] * self.y_symm[p,q] \
                                          + 4 * self.z_symm[p,q] * self.z_symm[p,q] \
                                          - ( self.x_symm[p,p] - self.x_symm[q,q] ) * ( self.x_symm[p,p] - self.x_symm[q,q] ) \
                                          - ( self.y_symm[p,p] - self.y_symm[q,q] ) * ( self.y_symm[p,p] - self.y_symm[q,q] ) \
                                          - ( self.z_symm[p,p] - self.z_symm[q,q] ) * ( self.z_symm[p,p] - self.z_symm[q,q] )
                    increment += 1
            diagonal *= -self.Norbs
        if ( self.__which == 'edmiston' ):
            increment = 0
            for p in range( self.Norbs ):
                for q in range( p+1, self.Norbs ):
                    diagonal[ increment ] = 16 * self.eri_rot[p,q,p,q] \
                                          +  8 * self.eri_rot[p,p,q,q] \
                                          -  4 * self.eri_rot[q,q,q,q] \
                                          -  4 * self.eri_rot[p,p,p,p]
                    increment += 1
            diagonal *= -1
        return diagonal
        
    
    def __debug_diag_hessian( self ):

        result1 = self.__diag_hessian()
        result2 = np.zeros( [ self.numVars ], dtype=float )
        
        for elem in range( self.numVars ):
            work = np.zeros( [ self.numVars ], dtype=float )
            work[ elem ] = 1.0
            matvec = self.__hessian_matvec( work )
            result2[ elem ] = matvec[ elem ]
            
        logger.debug(self, "2-norm( diag(hessian) difference ) = %g", np.linalg.norm( result1 - result2 ))
        logger.debug(self, "2-norm( diag(hessian) )            = %g", np.linalg.norm( result1 ))
    
    
    def __hessian_matvec( self, vecin ):
    
        vector_out = np.zeros( [ self.numVars ], dtype=ctypes.c_double )
        vector_inp = np.asarray( vecin, order='C', dtype=ctypes.c_double )
        
        if ( self.__which == 'boys' ):
        
            liblocalizer.hessian_boys( ctypes.c_int( self.Norbs ),
                                       self.x_symm.ctypes.data_as( ctypes.c_void_p ),
                                       self.y_symm.ctypes.data_as( ctypes.c_void_p ),
                                       self.z_symm.ctypes.data_as( ctypes.c_void_p ),
                                        vector_inp.ctypes.data_as( ctypes.c_void_p ),
                                        vector_out.ctypes.data_as( ctypes.c_void_p ) )
        
        if ( self.__which == 'edmiston' ):
        
            liblocalizer.hessian_edmiston( ctypes.c_int( self.Norbs ),
                                           self.eri_rot.ctypes.data_as( ctypes.c_void_p ),
                                             vector_inp.ctypes.data_as( ctypes.c_void_p ),
                                             vector_out.ctypes.data_as( ctypes.c_void_p ) )
        
        return vector_out
        
    
    def __debug_hessian_matvec( self ):

        hessian_analytic = np.zeros( [ self.numVars, self.numVars ], dtype=float )
        
        for cnt in range( self.numVars ):
            vector = np.zeros( [ self.numVars ], dtype=float )
            vector[ cnt ] = 1.0
            hessian_analytic[ :, cnt ] = self.__hessian_matvec( vector )

        original_umatrix = np.array( self.u, copy=True )

        stepsize = 1e-8
        self.__set_gradient()
        gradient_ref = np.array( self.gradient, copy=True )
        hessian_numerical = np.zeros( [ self.numVars, self.numVars ], dtype=float )
        for counter in range( self.numVars ):
            self.u = np.array( original_umatrix, copy=True )
            flatx = np.zeros( [ self.numVars ], dtype=float )
            flatx[counter] = stepsize
            self.__update_unitary( flatx )
            self.__set_gradient()
            hessian_numerical[ :, counter ] = ( self.gradient - gradient_ref ) / stepsize

        self.u = np.array( original_umatrix, copy=True )
        flatx = np.zeros( [ self.numVars ], dtype=float )
        self.__update_unitary( flatx )

        hessian_numerical = 0.5 * ( hessian_numerical + hessian_numerical.T )

        logger.debug(self, "2-norm( hessian difference ) = %g", np.linalg.norm( hessian_analytic - hessian_numerical ))
        logger.debug(self, "2-norm( hessian )            = %g", np.linalg.norm( hessian_analytic ))
    
    
    def __augmented_hessian_matvec( self, vector_in ):
    
        '''
            [ H    g ] [ v ]
            [ g^T  0 ] [ s ]
        '''
        #start_time = time.time()
        result = np.zeros( [ self.numVars + 1 ], dtype=float )
        result[ self.numVars ] = np.sum(np.multiply( self.gradient, vector_in[ :-1 ] ))
        result[ :-1 ] = self.__hessian_matvec( vector_in[ :-1 ] ) + vector_in[ self.numVars ] * self.gradient
        #self.ahnr_cnt += 1
        #end_time = time.time()
        #logger.debug(self, "Localizer :: Augmented Hessian matvec no. %d takes %g seconds.", self.ahnr_cnt, end_time-start_time)
        return result
    
    
    def __reorder_orbitals( self ):
    
        # Find the coordinates of each localized orbital (expectation value).
        __coords = np.zeros( [ self.Norbs, 3 ], dtype=float )
        if ( self.__which == 'boys' ):
            for orb in range( self.Norbs ):
                __coords[ orb, 0 ] = 0.5 * self.x_symm[ orb, orb ]
                __coords[ orb, 1 ] = 0.5 * self.y_symm[ orb, orb ]
                __coords[ orb, 2 ] = 0.5 * self.z_symm[ orb, orb ]
        if ( self.__which == 'edmiston' ):
            rvec     = self.themol.intor('cint1e_r_sph', 3)
            rotation = np.dot( self.coeff, self.u )
            for cart in range(3): #xyz
                __coords[ :, cart ] = np.diag( np.dot( np.dot( rotation.T, rvec[cart] ) , rotation ) )
                
        # Find the atom number to which the localized orbital is closest (RMS).
        __atomid = np.zeros( [ self.Norbs ], dtype=int )
        for orb in range( self.Norbs ):
            min_id = 0
            min_distance = np.linalg.norm( __coords[ orb, : ] - self.themol.atom_coord( 0 ) )
            for atom in range( 1, self.themol.natm ):
                current_distance = np.linalg.norm( __coords[ orb, : ] - self.themol.atom_coord( atom ) )
                if ( current_distance < min_distance ):
                    min_distance = current_distance
                    min_id = atom
            __atomid[ orb ] = min_id
            
        # Reorder
        idx = __atomid.argsort()
        self.u = self.u[:,idx]
    
    
    def optimize( self, threshold=1e-6 ):
        r'''Augmented Hessian Newton-Raphson optimization of the localization cost function, using an exact gradient and hessian
        
        Args:
            threshold : The convergence threshold for the orbital rotation gradient
            
        Returns:
            The orbital coefficients of the orthonormal localized orbitals, expressed in terms of the AO
        '''

        # To break up symmetrical orbitals
        flatx = ( 0.0123 / self.numVars ) * np.ones( [ self.numVars ], dtype=float )
        self.__update_unitary( flatx )

        #self.__debug_gradient()
        #self.__debug_hessian_matvec()
        #self.__debug_diag_hessian()

        self.grd_norm = 1.0
        iteration = 0
        max_cf_encountered = 0.0
        logger.debug(self, "Localizer :: At iteration %d the cost function = %1.13f", iteration, -self.__costfunction())
        logger.debug(self, "Localizer :: Linear size of the augmented Hessian = %d", self.numVars+1)

        while ( self.grd_norm > threshold ):

            iteration += 1
            self.__set_gradient() # Sets self.gradient and self.grd_norm
            
            ini_guess = np.zeros( [ self.numVars + 1 ], dtype=float )
            diag_h    = np.zeros( [ self.numVars + 1 ], dtype=float )
            ini_guess[ self.numVars ] = 1.0
            diag_h[ :-1 ] = self.__diag_hessian()
            for elem in range( self.numVars ):
                if ( abs( diag_h[ elem ] ) < 1e-6 ):
                    ini_guess[ elem ] = -self.gradient[ elem ] / 1e-6
                else:
                    ini_guess[ elem ] = -self.gradient[ elem ] / diag_h[ elem ] # Minus the gradient divided by the diagonal elements of the hessian
            if ( self.use_hess == True ):
            
                def myprecon( resid, eigval, eigvec ):
                
                    myprecon_cutoff = 1e-10
                    local_myprecon = np.zeros( [ self.numVars + 1 ], dtype=float )
                    for elem in range( self.numVars + 1 ):
                        if ( abs( diag_h[ elem ] - eigval ) < myprecon_cutoff ):
                            local_myprecon[ elem ] = eigvec[ elem ] / myprecon_cutoff
                        else:
                            # local_myprecon = eigvec / ( diag(H) - eigval ) = K^{-1} u
                            local_myprecon[ elem ] = eigvec[ elem ] / ( diag_h[ elem ] - eigval )
                    # alpha_myprecon = - ( r, K^{-1} u ) / ( u, K^{-1} u )
                    alpha_myprecon = - np.einsum( 'i,i->', local_myprecon, resid ) / np.einsum( 'i,i->', local_myprecon, eigvec )
                    # local_myprecon = r - ( r, K^{-1} u ) / ( u, K^{-1} u ) * u
                    local_myprecon = resid + alpha_myprecon * eigvec
                    for elem in range( self.numVars + 1 ):
                        if ( abs( diag_h[ elem ] - eigval ) < myprecon_cutoff ):
                            local_myprecon[ elem ] = - local_myprecon[ elem ] / myprecon_cutoff
                        else:
                            local_myprecon[ elem ] = - local_myprecon[ elem ] / ( diag_h[ elem ] - eigval )
                    return local_myprecon
                
                #self.ahnr_cnt = 0
                eigenval, eigenvec = linalg_helper.davidson( aop=self.__augmented_hessian_matvec, \
                                                             x0=ini_guess, \
                                                             precond=myprecon, \
                                                             #tol=1e-14, \
                                                             #max_cycle=50, \
                                                             max_space=20, \
                                                             #lindep=1e-16, \
                                                             #max_memory=2000, \
                                                             nroots=1 )
                                                             
            else:
                eigenvec = np.array( ini_guess, copy=True )
            flatx = eigenvec[ :-1 ] / eigenvec[ self.numVars ]

            update_norm = np.linalg.norm( flatx )
            cost_func_prev = -self.__costfunction()
            self.__update_unitary( flatx )
            cost_func_now = -self.__costfunction()
            counter = 0
            while ( counter < 8 ) and ( cost_func_now < cost_func_prev ):
                logger.debug(self, "Localizer :: Taking half a step back")
                flatx *= 0.5
                self.__update_unitary( -flatx )
                cost_func_now = -self.__costfunction()
                counter += 1
                
            if ( cost_func_now > max_cf_encountered ):
                max_cf_encountered = cost_func_now

            logger.debug(self, "Localizer :: Gradient norm = %g", self.grd_norm)
            logger.debug(self, "Localizer :: Update norm   = %g", update_norm)
            logger.debug(self, "Localizer :: At iteration %d the cost function = %1.13f", iteration, cost_func_now)
            logger.debug(self, "             Diff. with prev. CF = %g", cost_func_now - cost_func_prev )
            logger.debug(self, "             Diff. with max.  CF = %g", cost_func_now - max_cf_encountered )
            
            if ( iteration % 10 == 0 ):
                self.__reorthogonalize()
                cost_func_now = -self.__costfunction()

        logger.note(self, "Localization procedure converged in %d iterations.", iteration)
        
        self.__reorder_orbitals()
        converged_coeff = np.dot( self.coeff, self.u )
        return converged_coeff

