/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
  
   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
 
        http://www.apache.org/licenses/LICENSE-2.0
 
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Author: Sebastian Wouters <sebastianwouters@gmail.com>
 *
 *  Date: August 3, 2015
 *
 *  Augmented Hessian Newton-Raphson optimization of
 *     1. either the Edmiston-Ruedenberg localization cost function
 *     2. or the Boys localization cost function
 *  in both cases with an analytic gradient and hessian
 *
 *  Reference: C. Edmiston and K. Ruedenberg, Reviews of Modern Physics 35, 457-464 (1963). http://dx.doi.org/10.1103/RevModPhys.35.457
 *             http://sebwouters.github.io/CheMPS2/doxygen/classCheMPS2_1_1EdmistonRuedenberg.html
 */

#include <stdlib.h>

void hessian_boys(const int Norbs, double * x_symm, double * y_symm, double * z_symm, double * vector_in, double * vector_out){

    int rowindex,p,q,r,s;
    const int num_vars = (Norbs*(Norbs-1))/2;
    
//    for (p=0; p<Norbs; p++){
//        for (q=p+1; q<Norbs; q++){
//            const int rowindex == q + p * Norbs - ((p+1)*(p+2))/2
    
    #pragma omp parallel for schedule(static)
    for (rowindex=0; rowindex<num_vars; rowindex++){
    
        s = num_vars - 1 - rowindex;
        r = 2;
        while ( (r*(r-1))/2 <= s ){ r++; }
        p = Norbs - r;
        q = rowindex - p*Norbs + ((p+1)*(p+2))/2;
        
        double value = 0.0;
        
        // Part 1: p == r
        for (s=p+1; s<Norbs; s++){
            const int arg_pq = p + Norbs * q;
            const int arg_ps = p + Norbs * s;
            const int arg_qs = q + Norbs * s;
            const int arg_pp = p + Norbs * p;
            const int arg_qq = q + Norbs * q;
            const int arg_ss = s + Norbs * s;
            const double prefactor = ( 2 * x_symm[arg_pq] * x_symm[arg_ps] + x_symm[arg_qs] * ( x_symm[arg_pp] - 0.5 * x_symm[arg_qq] - 0.5 * x_symm[arg_ss] ) )
                                   + ( 2 * y_symm[arg_pq] * y_symm[arg_ps] + y_symm[arg_qs] * ( y_symm[arg_pp] - 0.5 * y_symm[arg_qq] - 0.5 * y_symm[arg_ss] ) )
                                   + ( 2 * z_symm[arg_pq] * z_symm[arg_ps] + z_symm[arg_qs] * ( z_symm[arg_pp] - 0.5 * z_symm[arg_qq] - 0.5 * z_symm[arg_ss] ) );
            
            const int colindex = rowindex + s - q; // s + p * Norbs - ((p+1)*(p+2))/2
            value += prefactor * vector_in[ colindex ];
        }
        
        // Part 2: q == s
        for (r=0; r<q; r++){
            const int arg_pq = p + Norbs * q;
            const int arg_rq = r + Norbs * q;
            const int arg_pr = p + Norbs * r;
            const int arg_qq = q + Norbs * q;
            const int arg_pp = p + Norbs * p;
            const int arg_rr = r + Norbs * r;
            const double prefactor = ( 2 * x_symm[arg_pq] * x_symm[arg_rq] + x_symm[arg_pr] * ( x_symm[arg_qq] - 0.5 * x_symm[arg_pp] - 0.5 * x_symm[arg_rr] ) )
                                   + ( 2 * y_symm[arg_pq] * y_symm[arg_rq] + y_symm[arg_pr] * ( y_symm[arg_qq] - 0.5 * y_symm[arg_pp] - 0.5 * y_symm[arg_rr] ) )
                                   + ( 2 * z_symm[arg_pq] * z_symm[arg_rq] + z_symm[arg_pr] * ( z_symm[arg_qq] - 0.5 * z_symm[arg_pp] - 0.5 * z_symm[arg_rr] ) );
            
            const int colindex = q + r * Norbs - ((r+1)*(r+2))/2;
            value += prefactor * vector_in[ colindex ];
        }
        
        // Part 3: q == r
        for (s=q+1; s<Norbs; s++){
            const int arg_pq = p + Norbs * q;
            const int arg_qs = q + Norbs * s;
            const int arg_ps = p + Norbs * s;
            const int arg_qq = q + Norbs * q;
            const int arg_pp = p + Norbs * p;
            const int arg_ss = s + Norbs * s;
            const double prefactor = ( 2 * x_symm[arg_pq] * x_symm[arg_qs] + x_symm[arg_ps] * ( x_symm[arg_qq] - 0.5 * x_symm[arg_pp] - 0.5 * x_symm[arg_ss] ) )
                                   + ( 2 * y_symm[arg_pq] * y_symm[arg_qs] + y_symm[arg_ps] * ( y_symm[arg_qq] - 0.5 * y_symm[arg_pp] - 0.5 * y_symm[arg_ss] ) )
                                   + ( 2 * z_symm[arg_pq] * z_symm[arg_qs] + z_symm[arg_ps] * ( z_symm[arg_qq] - 0.5 * z_symm[arg_pp] - 0.5 * z_symm[arg_ss] ) );
            
            const int colindex = s + q * Norbs - ((q+1)*(q+2))/2;
            value -= prefactor * vector_in[ colindex ];
        }
        
        // Part 4: p == s
        for (r=0; r<p; r++){
            const int arg_pq = p + Norbs * q;
            const int arg_rp = r + Norbs * p;
            const int arg_qr = q + Norbs * r;
            const int arg_pp = p + Norbs * p;
            const int arg_qq = q + Norbs * q;
            const int arg_rr = r + Norbs * r;
            const double prefactor = ( 2 * x_symm[arg_pq] * x_symm[arg_rp] + x_symm[arg_qr] * ( x_symm[arg_pp] - 0.5 * x_symm[arg_qq] - 0.5 * x_symm[arg_rr] ) )
                                   + ( 2 * y_symm[arg_pq] * y_symm[arg_rp] + y_symm[arg_qr] * ( y_symm[arg_pp] - 0.5 * y_symm[arg_qq] - 0.5 * y_symm[arg_rr] ) )
                                   + ( 2 * z_symm[arg_pq] * z_symm[arg_rp] + z_symm[arg_qr] * ( z_symm[arg_pp] - 0.5 * z_symm[arg_qq] - 0.5 * z_symm[arg_rr] ) );
            
            const int colindex = p + r * Norbs - ((r+1)*(r+2))/2;
            value -= prefactor * vector_in[ colindex ];
        }
        
        vector_out[ rowindex ] = -Norbs*value;
        
    }

}

void hessian_edmiston(const int Norbs, double * eri, double * vector_in, double * vector_out){

    int rowindex,p,q,r,s;
    const int num_vars = (Norbs*(Norbs-1))/2;
    
//    for (p=0; p<Norbs; p++){
//        for (q=p+1; q<Norbs; q++){
//            const int rowindex == q + p * Norbs - ((p+1)*(p+2))/2
    
    #pragma omp parallel for schedule(static)
    for (rowindex=0; rowindex<num_vars; rowindex++){
    
        s = num_vars - 1 - rowindex;
        r = 2;
        while ( (r*(r-1))/2 <= s ){ r++; }
        p = Norbs - r;
        q = rowindex - p*Norbs + ((p+1)*(p+2))/2;
        
        double value = 0.0;
        
        // Part 1: p == r
        for (s=p+1; s<Norbs; s++){
            const int pqps = p + Norbs * ( q + Norbs * ( p + Norbs * s ));
            const int ppqs = p + Norbs * ( p + Norbs * ( q + Norbs * s ));
            const int qqqs = q + Norbs * ( q + Norbs * ( q + Norbs * s ));
            const int sssq = s + Norbs * ( s + Norbs * ( s + Norbs * q ));
            const double prefactor = 2*(4*eri[pqps] + 2*eri[ppqs] - eri[qqqs] - eri[sssq]);
            const int colindex = rowindex + s - q; // s + p * Norbs - ((p+1)*(p+2))/2
            value += prefactor * vector_in[ colindex ];
        }
        
        // Part 2: q == s
        for (r=0; r<q; r++){
            const int qpqr = q + Norbs * ( p + Norbs * ( q + Norbs * r ));
            const int qqpr = q + Norbs * ( q + Norbs * ( p + Norbs * r ));
            const int pppr = p + Norbs * ( p + Norbs * ( p + Norbs * r ));
            const int rrrp = r + Norbs * ( r + Norbs * ( r + Norbs * p ));
            const double prefactor = 2*(4*eri[qpqr] + 2*eri[qqpr] - eri[pppr] - eri[rrrp]);
            const int colindex = q + r * Norbs - ((r+1)*(r+2))/2;
            value += prefactor * vector_in[ colindex ];
        }
        
        // Part 3: q == r
        for (s=q+1; s<Norbs; s++){
            const int qpqs = q + Norbs * ( p + Norbs * ( q + Norbs * s ));
            const int qqps = q + Norbs * ( q + Norbs * ( p + Norbs * s ));
            const int ppps = p + Norbs * ( p + Norbs * ( p + Norbs * s ));
            const int sssp = s + Norbs * ( s + Norbs * ( s + Norbs * p ));
            const double prefactor = 2*(4*eri[qpqs] + 2*eri[qqps] - eri[ppps] - eri[sssp]);
            const int colindex = s + q * Norbs - ((q+1)*(q+2))/2;
            value -= prefactor * vector_in[ colindex ];
        }
        
        // Part 4: p == s
        for (r=0; r<p; r++){
            const int pqpr = p + Norbs * ( q + Norbs * ( p + Norbs * r ));
            const int ppqr = p + Norbs * ( p + Norbs * ( q + Norbs * r ));
            const int qqqr = q + Norbs * ( q + Norbs * ( q + Norbs * r ));
            const int rrrq = r + Norbs * ( r + Norbs * ( r + Norbs * q ));
            const double prefactor = 2*(4*eri[pqpr] + 2*eri[ppqr] - eri[qqqr] - eri[rrrq]);
            const int colindex = p + r * Norbs - ((r+1)*(r+2))/2;
            value -= prefactor * vector_in[ colindex ];
        }
        
        vector_out[ rowindex ] = -value;
        
    }

}

