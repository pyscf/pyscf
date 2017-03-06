/*
 *  C functions for Heat-Bath CI implementation
 */
#include <stdint.h>
#define MAX_THREADS     256

void contract_h_c(double *h1, double *eri, int norb, int neleca, int nelecb, uint64_t *strs, double *civec, double *hdiag, int ndet, double *ci1);
int n_excitations(uint64_t *str1, uint64_t *str2, int nset);
int popcount(uint64_t bb);
int *get_single_excitation(uint64_t *str1, uint64_t *str2, int nset);
int *get_double_excitation(uint64_t *str1, uint64_t *str2, int nset);
int trailz(uint64_t v);
char *int2bin(int i);
double compute_cre_des_sign(int a, int i, uint64_t *stria, int nset);
int *compute_occ_list(uint64_t *string, int nset, int norb, int nelec);
