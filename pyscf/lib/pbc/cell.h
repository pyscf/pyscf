#define RCUT_MAX_CYCLE 100

double pgf_rcut(int l, double alpha, double coeff,
                double precision, double r0, int max_cycle);

void rcut_by_shells(double* shell_radius, double** ptr_pgf_rcut,
                    int* bas, double* env, int nbas,
                    double r0, double precision);
