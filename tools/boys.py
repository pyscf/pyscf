#!/usr/bin/env python
#
# Author: Sebastian Wouters <sebastianwouters@gmail.com>
#
# Date: March 4, 2015
#
# Augmented Hessian Newton-Raphson optimization of the Boys
# localization cost function, with an exact gradient and hessian
#

from pyscf import gto, scf
from pyscf.tools import molden
from pyscf.lib import parameters as param

import numpy as np
import scipy

class boys:

    def __init__( self, mol, orbital_coeff ):
        r'''Initializer for the boys localization procedure

        Args:
            mol : A molecule which has been built
            orbital_coeff: Set of orthonormal orbitals, expressed in terms of the AO, which should be localized
        '''
    
        self.themol  = mol
        self.coeff   = orbital_coeff
        self.Norbs   = orbital_coeff.shape[1]
        self.numVars = ( self.Norbs * ( self.Norbs - 1 ) ) / 2
        
        rvec        = self.themol.intor('cint1e_r_sph', 3)
        self.x_orig = np.dot( np.dot( self.coeff.T, rvec[0] ) , self.coeff )
        self.y_orig = np.dot( np.dot( self.coeff.T, rvec[1] ) , self.coeff )
        self.z_orig = np.dot( np.dot( self.coeff.T, rvec[2] ) , self.coeff )
        
        self.u      = np.eye( self.Norbs, dtype=float )
        self.x_symm = self.x_orig + self.x_orig.T
        self.y_symm = self.x_orig + self.x_orig.T
        self.z_symm = self.x_orig + self.x_orig.T
        
        self.verbose = param.VERBOSE_NOTICE
        
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
        x_curr = np.dot( np.dot( self.u.T, self.x_orig ), self.u )
        y_curr = np.dot( np.dot( self.u.T, self.y_orig ), self.u )
        z_curr = np.dot( np.dot( self.u.T, self.z_orig ), self.u )
        self.x_symm = x_curr + x_curr.T
        self.y_symm = y_curr + y_curr.T
        self.z_symm = z_curr + z_curr.T
        
    def __costfunction( self ):
    
        value = 0.0
        for i in range( self.Norbs ):
            for j in range( i+1, self.Norbs ): # j > i
                temp = self.x_symm[i,i] - self.x_symm[j,j]
                value += temp * temp
                temp = self.y_symm[i,i] - self.y_symm[j,j]
                value += temp * temp
                temp = self.z_symm[i,i] - self.z_symm[j,j]
                value += temp * temp
        value *= -0.25
        return value
        
    def __gradient( self ):
        
        grad = np.zeros( [ self.numVars ], dtype=float )
        increment = 0
        for p in range( self.Norbs ):
            for q in range( p+1, self.Norbs ):
                grad[ increment ] = self.x_symm[p,q] * ( self.x_symm[q,q] - self.x_symm[p,p] ) \
                                  + self.y_symm[p,q] * ( self.y_symm[q,q] - self.y_symm[p,p] ) \
                                  + self.z_symm[p,q] * ( self.z_symm[q,q] - self.z_symm[p,p] )
                increment += 1
        grad *= -self.Norbs
        return grad
        
    def __debug_gradient( self ):
    
        gradient_analytic = self.__gradient()
        
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
        
        print "2-norm( gradient difference ) =", np.linalg.norm( gradient_analytic - gradient_numerical )
        print "2-norm( gradient )            =", np.linalg.norm( gradient_analytic )

    def __hessian( self ):
    
        hess = np.zeros( [ self.numVars, self.numVars ], dtype=float )
        increment_col = 0
        for r in range( self.Norbs ):
            for s in range( r+1, self.Norbs ):
                increment_row = 0
                for p in range( self.Norbs ):
                    for q in range( p+1, self.Norbs ):
                        if ( increment_row <= increment_col ):
                            prefactor = 0
                            value = 0.0
                            if ( p == r ):
                                prefactor += 1
                                value += ( self.x_symm[q,s] * ( self.x_symm[p,p] - 0.5 * self.x_symm[q,q] - 0.5 * self.x_symm[s,s] ) \
                                         + self.y_symm[q,s] * ( self.y_symm[p,p] - 0.5 * self.y_symm[q,q] - 0.5 * self.y_symm[s,s] ) \
                                         + self.z_symm[q,s] * ( self.z_symm[p,p] - 0.5 * self.z_symm[q,q] - 0.5 * self.z_symm[s,s] ) )
                            if ( q == s ):
                                prefactor += 1
                                value += ( self.x_symm[p,r] * ( self.x_symm[q,q] - 0.5 * self.x_symm[p,p] - 0.5 * self.x_symm[r,r] ) \
                                         + self.y_symm[p,r] * ( self.y_symm[q,q] - 0.5 * self.y_symm[p,p] - 0.5 * self.y_symm[r,r] ) \
                                         + self.z_symm[p,r] * ( self.z_symm[q,q] - 0.5 * self.z_symm[p,p] - 0.5 * self.z_symm[r,r] ) )
                            if ( q == r ):
                                prefactor -= 1
                                value -= ( self.x_symm[p,s] * ( self.x_symm[q,q] - 0.5 * self.x_symm[p,p] - 0.5 * self.x_symm[s,s] ) \
                                         + self.y_symm[p,s] * ( self.y_symm[q,q] - 0.5 * self.y_symm[p,p] - 0.5 * self.y_symm[s,s] ) \
                                         + self.z_symm[p,s] * ( self.z_symm[q,q] - 0.5 * self.z_symm[p,p] - 0.5 * self.z_symm[s,s] ) )
                            if ( p == s ):
                                prefactor -= 1
                                value -= ( self.x_symm[q,r] * ( self.x_symm[p,p] - 0.5 * self.x_symm[r,r] - 0.5 * self.x_symm[q,q] ) \
                                         + self.y_symm[q,r] * ( self.y_symm[p,p] - 0.5 * self.y_symm[r,r] - 0.5 * self.y_symm[q,q] ) \
                                         + self.z_symm[q,r] * ( self.z_symm[p,p] - 0.5 * self.z_symm[r,r] - 0.5 * self.z_symm[q,q] ) )
                            if ( prefactor != 0 ):
                                value += 2 * prefactor * ( self.x_symm[p,q] * self.x_symm[r,s] + self.y_symm[p,q] * self.y_symm[r,s] + self.z_symm[p,q] * self.z_symm[r,s] )
                            hess[ increment_row, increment_col ] = value
                            hess[ increment_col, increment_row ] = value
                        increment_row += 1
                increment_col += 1
        hess *= -self.Norbs
        return hess
        
    def __debug_hessian( self ):
    
        hessian_analytic = self.__hessian()
        
        original_umatrix = np.array( self.u, copy=True )
        
        stepsize = 1e-8
        gradient_ref = self.__gradient()
        hessian_numerical = np.zeros( [ self.numVars, self.numVars ], dtype=float )
        for counter in range( self.numVars ):
            self.u = np.array( original_umatrix, copy=True )
            flatx = np.zeros( [ self.numVars ], dtype=float )
            flatx[counter] = stepsize
            self.__update_unitary( flatx )
            gradient_step = self.__gradient()
            hessian_numerical[ :, counter ] = ( gradient_step - gradient_ref ) / stepsize
        
        self.u = np.array( original_umatrix, copy=True )
        flatx = np.zeros( [ self.numVars ], dtype=float )
        self.__update_unitary( flatx )
        
        hessian_numerical = 0.5 * ( hessian_numerical + hessian_numerical.T )
        
        print "2-norm( hessian difference ) =", np.linalg.norm( hessian_analytic - hessian_numerical )
        print "2-norm( hessian )            =", np.linalg.norm( hessian_analytic )

    def optimize( self ):
        r'''Augmented Hessian Newton-Raphson optimization of the Boys localization cost function, using an exact gradient and hessian
        
        Returns:
            The orbital coefficients of the orthonormal localized orbitals, expressed in terms of the AO
        '''
    
        # To break up symmetrical orbitals
        flatx = 0.0123 * np.ones( [ self.numVars ], dtype=float )
        self.__update_unitary( flatx )
        
        #self.__debug_gradient()
        #self.__debug_hessian()
        
        gradient_norm = 1.0
        threshold = 1e-6
        iteration = 0
        if ( self.verbose >= param.VERBOSE_DEBUG ):
            print "Boys :: At iteration", iteration, "the cost function =", -self.__costfunction()
        
        while ( gradient_norm > threshold ):
        
            iteration += 1
            augmented = np.zeros( [ self.numVars+1, self.numVars+1 ], dtype=float )
            gradient = self.__gradient()
            augmented[:-1,:-1] = self.__hessian()
            augmented[:-1,self.numVars] = gradient
            augmented[self.numVars,:-1] = gradient
            eigenvals, eigenvecs = np.linalg.eigh( augmented )
            idx = eigenvals.argsort()
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:,idx]
            flatx = eigenvecs[:-1,0] / eigenvecs[self.numVars,0]
            
            gradient_norm = np.linalg.norm( gradient )
            update_norm = np.linalg.norm( flatx )
            self.__update_unitary( flatx )
            
            if ( self.verbose >= param.VERBOSE_DEBUG ):
                print "Boys :: gradient norm =", gradient_norm
                print "Boys :: update norm   =", update_norm
                print "Boys :: At iteration", iteration, "the cost function =", -self.__costfunction()
                
        if ( self.verbose >= param.VERBOSE_NOTICE ):
            print "Boys localization procedure converged in", iteration, "iterations."
            
        converged_coeff = np.dot( self.coeff, self.u )
        return converged_coeff
        
