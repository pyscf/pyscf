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

#ifndef HAVE_DEFINED_NEIGHBOR_LIST_H
#define HAVE_DEFINED_NEIGHBOR_LIST_H
typedef struct NeighborPair_struct {
    int nimgs;
    int *Ls_list;
    double *q_cond;
    double *center;
} NeighborPair;

typedef struct NeighborList_struct {
    int nish;
    int njsh;
    int nimgs;
    NeighborPair **pairs;
} NeighborList;

typedef struct NeighborListOpt_struct {
    NeighborList *nl;
    int (*fprescreen)(int *shls, struct NeighborListOpt_struct *opt);
} NeighborListOpt;

int NLOpt_noscreen(int* shls, NeighborListOpt* opt);
#endif
