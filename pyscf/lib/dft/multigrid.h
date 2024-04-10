/* Copyright 2021- The PySCF Developers. All Rights Reserved.

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
 * Author: Xing Zhang <zhangxing.nju@gmail.com>
 */

#ifndef HAVE_DEFINED_MULTIGRID_H
#define HAVE_DEFINED_MULTIGRID_H

#include <stdbool.h>

#define BINOMIAL(n, i)  (_BINOMIAL_COEF[_LEN_CART0[n]+i])

extern const int _LEN_CART[];
extern const int _LEN_CART0[];
extern const int _BINOMIAL_COEF[];

typedef struct GridLevel_Info_struct {
    int nlevels;
    double rel_cutoff;
    double *cutoff;
    int *mesh;
} GridLevel_Info;

typedef struct RS_Grid_struct {
    int nlevels;
    GridLevel_Info* gridlevel_info;
    int comp;
    double** data;
} RS_Grid;

typedef struct PGFPair_struct {
    int ish;
    int ipgf;
    int jsh;
    int jpgf;
    int iL;
    double radius;
} PGFPair;

bool pgfpairs_with_same_shells(PGFPair*, PGFPair*);

typedef struct Task_struct {
    size_t buf_size;
    size_t ntasks;
    PGFPair** pgfpairs;
    double radius;
} Task;

typedef struct TaskList_struct {
    int nlevels;
    int hermi;
    GridLevel_Info* gridlevel_info;
    Task** tasks;
} TaskList;


int get_task_loc(int** task_loc, PGFPair** pgfpairs, int ntasks,
                 int ish0, int ish1, int jsh0, int jsh1, int hermi);
#endif
