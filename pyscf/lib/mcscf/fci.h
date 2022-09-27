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

#include <stdint.h>
#define MAX_THREADS     256

typedef struct {
        unsigned int addr;
        unsigned short ia;
        int8_t sign;
        int8_t _padding;
} _LinkTrilT;

typedef struct {
        unsigned int addr;
        uint8_t a;
        uint8_t i;
        int8_t sign;
        int8_t _padding;
} _LinkT;

#define EXTRACT_A(I)    (I.a)
#define EXTRACT_I(I)    (I.i)
#define EXTRACT_SIGN(I) (I.sign)
#define EXTRACT_ADDR(I) (I.addr)
#define EXTRACT_IA(I)   (I.ia)

#define EXTRACT_CRE(I)  EXTRACT_A(I)
#define EXTRACT_DES(I)  EXTRACT_I(I)

void FCIcompress_link(_LinkT *clink, int *link_index,
                      int norb, int nstr, int nlink);
void FCIcompress_link_tril(_LinkTrilT *clink, int *link_index,
                           int nstr, int nlink);
int FCIcre_des_sign(int p, int q, uint64_t string0);
int FCIcre_sign(int p, uint64_t string0);
int FCIdes_sign(int p, uint64_t string0);
int FCIpopcount_1(uint64_t x);

void FCIprog_a_t1(double *ci0, double *t1,
                  int bcount, int stra_id, int strb_id,
                  int norb, int nstrb, int nlinka, _LinkTrilT *clink_indexa);
void FCIprog_b_t1(double *ci0, double *t1,
                  int bcount, int stra_id, int strb_id,
                  int norb, int nstrb, int nlinka, _LinkTrilT *clink_indexa);
void FCIspread_a_t1(double *ci0, double *t1,
                    int bcount, int stra_id, int strb_id,
                    int norb, int nstrb, int nlinka, _LinkTrilT *clink_indexa);
void FCIspread_b_t1(double *ci0, double *t1,
                    int bcount, int stra_id, int strb_id,
                    int norb, int nstrb, int nlinka, _LinkTrilT *clink_indexa);

double FCIrdm2_a_t1ci(double *ci0, double *t1,
                      int bcount, int stra_id, int strb_id,
                      int norb, int nstrb, int nlinka, _LinkT *clink_indexa);
double FCIrdm2_b_t1ci(double *ci0, double *t1,
                      int bcount, int stra_id, int strb_id,
                      int norb, int nstrb, int nlinka, _LinkT *clink_indexa);

void FCIaxpy2d(double *out, double *in, size_t count, size_t no, size_t ni);
