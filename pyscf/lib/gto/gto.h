/* Copyright 2021 The PySCF Developers. All Rights Reserved.
  
   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
 
        http://www.apache.org/licenses/LICENSE-2.0
 
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
 */

int GTOmax_shell_dim(const int *ao_loc, const int *shls, int ncenter);
size_t GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                         int *atm, int natm, int *bas, int nbas, double *env);
