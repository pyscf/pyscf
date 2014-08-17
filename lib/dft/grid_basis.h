/*
 * File: grid_basis.h
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

void VXCvalue_nr_gto(int nao, int ngrids, double *ao, double *coord,
                     int *atm, int natm, int *bas, int nbas, double *env);
void VXCvalue_nr_gto_grad(int nao, int ngrids, double *ao, double *coord,
                          int *atm, int natm, int *bas, int nbas, double *env);
