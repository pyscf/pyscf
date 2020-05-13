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
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

void CVHFrs1_ji_s1kl(double complex *eri,
                     double complex *dm, double complex *vj,
                     int nao, int ncomp, int *shls, int *ao_loc, int *tao,
                       double *dm_cond, int nbas, double dm_atleast);
void CVHFrs1_lk_s1ij(double complex *eri,
                     double complex *dm, double complex *vj,
                     int nao, int ncomp, int *shls, int *ao_loc, int *tao,
                       double *dm_cond, int nbas, double dm_atleast);
void CVHFrs1_jk_s1il(double complex *eri,
                     double complex *dm, double complex *vk,
                     int nao, int ncomp, int *shls, int *ao_loc, int *tao,
                       double *dm_cond, int nbas, double dm_atleast);
void CVHFrs1_li_s1kj(double complex *eri,
                     double complex *dm, double complex *vk,
                     int nao, int ncomp, int *shls, int *ao_loc, int *tao,
                       double *dm_cond, int nbas, double dm_atleast);
void CVHFrs2ij_ji_s1kl(double complex *eri,
                       double complex *dm, double complex *vj,
                       int nao, int ncomp, int *shls, int *ao_loc, int *tao,
                       double *dm_cond, int nbas, double dm_atleast);
void CVHFrs2ij_lk_s2ij(double complex *eri,
                       double complex *dm, double complex *vj,
                       int nao, int ncomp, int *shls, int *ao_loc, int *tao,
                       double *dm_cond, int nbas, double dm_atleast);
void CVHFrs2ij_jk_s1il(double complex *eri,
                       double complex *dm, double complex *vk,
                       int nao, int ncomp, int *shls, int *ao_loc, int *tao,
                       double *dm_cond, int nbas, double dm_atleast);
void CVHFrs2ij_li_s1kj(double complex *eri,
                       double complex *dm, double complex *vk,
                       int nao, int ncomp, int *shls, int *ao_loc, int *tao,
                       double *dm_cond, int nbas, double dm_atleast);
void CVHFrs2kl_ji_s2kl(double complex *eri,
                       double complex *dm, double complex *vj,
                       int nao, int ncomp, int *shls, int *ao_loc, int *tao,
                       double *dm_cond, int nbas, double dm_atleast);
void CVHFrs2kl_lk_s1ij(double complex *eri,
                       double complex *dm, double complex *vj,
                       int nao, int ncomp, int *shls, int *ao_loc, int *tao,
                       double *dm_cond, int nbas, double dm_atleast);
void CVHFrs2kl_jk_s1il(double complex *eri,
                       double complex *dm, double complex *vk,
                       int nao, int ncomp, int *shls, int *ao_loc, int *tao,
                       double *dm_cond, int nbas, double dm_atleast);
void CVHFrs2kl_li_s1kj(double complex *eri,
                       double complex *dm, double complex *vk,
                       int nao, int ncomp, int *shls, int *ao_loc, int *tao,
                       double *dm_cond, int nbas, double dm_atleast);
void CVHFrs4_ji_s2kl(double complex *eri,
                     double complex *dm, double complex *vj,
                     int nao, int ncomp, int *shls, int *ao_loc, int *tao,
                     double *dm_cond, int nbas, double dm_atleast);
void CVHFrs4_lk_s2ij(double complex *eri,
                     double complex *dm, double complex *vj,
                     int nao, int ncomp, int *shls, int *ao_loc, int *tao,
                     double *dm_cond, int nbas, double dm_atleast);
void CVHFrs4_jk_s1il(double complex *eri,
                     double complex *dm, double complex *vk,
                     int nao, int ncomp, int *shls, int *ao_loc, int *tao,
                     double *dm_cond, int nbas, double dm_atleast);
void CVHFrs4_li_s1kj(double complex *eri,
                     double complex *dm, double complex *vk,
                     int nao, int ncomp, int *shls, int *ao_loc, int *tao,
                     double *dm_cond, int nbas, double dm_atleast);
void CVHFrs8_ji_s2kl(double complex *eri,
                     double complex *dm, double complex *vj,
                     int nao, int ncomp, int *shls, int *ao_loc, int *tao,
                     double *dm_cond, int nbas, double dm_atleast);
void CVHFrs8_lk_s2ij(double complex *eri,
                     double complex *dm, double complex *vj,
                     int nao, int ncomp, int *shls, int *ao_loc, int *tao,
                     double *dm_cond, int nbas, double dm_atleast);
void CVHFrs8_jk_s1il(double complex *eri,
                     double complex *dm, double complex *vk,
                     int nao, int ncomp, int *shls, int *ao_loc, int *tao,
                     double *dm_cond, int nbas, double dm_atleast);
void CVHFrs8_li_s1kj(double complex *eri,
                     double complex *dm, double complex *vk,
                     int nao, int ncomp, int *shls, int *ao_loc, int *tao,
                     double *dm_cond, int nbas, double dm_atleast);

