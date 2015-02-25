/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

void VXCeval_nr_gto(int nao, int ngrids, int blksize,
                    int bastart, int bascount, double *ao, double *coord,
                    char *non0table,
                    int *atm, int natm, int *bas, int nbas, double *env);

void VXCeval_nr_gto_grad(int nao, int ngrids, int blksize,
                         int bastart, int bascount, double *ao, double *coord,
                         char *non0table,
                         int *atm, int natm, int *bas, int nbas, double *env);

void VXCeval_ao_drv(void (*eval_gto)(),
                    int nao, int ngrids, int bastart, int bascount, int blksize,
                    double *ao, double *coord, char *non0table,
                    int *atm, int natm, int *bas, int nbas, double *env);

void VXCnr_ao_screen(char *non0table, double *coord, int ngrids, int blksize,
                     int *atm, int natm, int *bas, int nbas, double *env);
