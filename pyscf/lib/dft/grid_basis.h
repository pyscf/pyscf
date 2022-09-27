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

void VXCeval_ao_drv(int deriv, int nao, int ngrids,
                    int bastart, int bascount, int blksize,
                    double *ao, double *coord, uint8_t *non0table,
                    int *atm, int natm, int *bas, int nbas, double *env);

void VXCnr_ao_screen(uint8_t *non0table, double *coord, int ngrids, int blksize,
                     int *atm, int natm, int *bas, int nbas, double *env);
