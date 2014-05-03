
int cint1e_ovlp_sph(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_ovlp_sph(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_ovlp_sph", 1, &cint1e_ovlp_sph, NPY_DOUBLE, args);
}

int cint1e_kin_sph(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_kin_sph(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_kin_sph", 1, &cint1e_kin_sph, NPY_DOUBLE, args);
}

int cint1e_nuc_sph(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_nuc_sph(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_nuc_sph", 1, &cint1e_nuc_sph, NPY_DOUBLE, args);
}

int cint1e_giao_irjxp_sph(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_giao_irjxp_sph(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_giao_irjxp_sph", 3, &cint1e_giao_irjxp_sph, NPY_DOUBLE, args);
}

int cint1e_cg_irxp_sph(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_cg_irxp_sph(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_cg_irxp_sph", 3, &cint1e_cg_irxp_sph, NPY_DOUBLE, args);
}

int cint1e_ia01p_sph(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_ia01p_sph(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_ia01p_sph", 3, &cint1e_ia01p_sph, NPY_DOUBLE, args);
}

int cint1e_giao_a11part_sph(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_giao_a11part_sph(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_giao_a11part_sph", 9, &cint1e_giao_a11part_sph, NPY_DOUBLE, args);
}

int cint1e_cg_a11part_sph(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_cg_a11part_sph(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_cg_a11part_sph", 9, &cint1e_cg_a11part_sph, NPY_DOUBLE, args);
}

int cint1e_igovlp_sph(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_igovlp_sph(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_igovlp_sph", 3, &cint1e_igovlp_sph, NPY_DOUBLE, args);
}

int cint1e_igkin_sph(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_igkin_sph(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_igkin_sph", 3, &cint1e_igkin_sph, NPY_DOUBLE, args);
}

int cint1e_ignuc_sph(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_ignuc_sph(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_ignuc_sph", 3, &cint1e_ignuc_sph, NPY_DOUBLE, args);
}

int cint1e_a01gp_sph(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_a01gp_sph(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_a01gp_sph", 9, &cint1e_a01gp_sph, NPY_DOUBLE, args);
}

int cint1e_ipovlp_sph(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_ipovlp_sph(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_ipovlp_sph", 3, &cint1e_ipovlp_sph, NPY_DOUBLE, args);
}

int cint1e_ipkin_sph(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_ipkin_sph(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_ipkin_sph", 3, &cint1e_ipkin_sph, NPY_DOUBLE, args);
}

int cint1e_ipnuc_sph(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_ipnuc_sph(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_ipnuc_sph", 3, &cint1e_ipnuc_sph, NPY_DOUBLE, args);
}

int cint1e_iprinv_sph(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_iprinv_sph(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_iprinv_sph", 3, &cint1e_iprinv_sph, NPY_DOUBLE, args);
}

int cint1e_ovlp(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_ovlp(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_ovlp", 1, &cint1e_ovlp, NPY_COMPLEX, args);
}

int cint1e_spsp(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_spsp(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_spsp", 1, &cint1e_spsp, NPY_COMPLEX, args);
}

int cint1e_nuc(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_nuc(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_nuc", 1, &cint1e_nuc, NPY_COMPLEX, args);
}

int cint1e_spnucsp(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_spnucsp(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_spnucsp", 1, &cint1e_spnucsp, NPY_COMPLEX, args);
}

int cint1e_srnucsr(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_srnucsr(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_srnucsr", 1, &cint1e_srnucsr, NPY_COMPLEX, args);
}

int cint1e_sp(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_sp(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_sp", 1, &cint1e_sp, NPY_COMPLEX, args);
}

int cint1e_srsr(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_srsr(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_srsr", 1, &cint1e_srsr, NPY_COMPLEX, args);
}

int cint1e_srsp(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_srsp(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_srsp", 1, &cint1e_srsp, NPY_COMPLEX, args);
}

int cint1e_spspsp(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_spspsp(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_spspsp", 1, &cint1e_spspsp, NPY_COMPLEX, args);
}

int cint1e_spnuc(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_spnuc(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_spnuc", 1, &cint1e_spnuc, NPY_COMPLEX, args);
}

int cint1e_ipovlp(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_ipovlp(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_ipovlp", 3, &cint1e_ipovlp, NPY_COMPLEX, args);
}

int cint1e_ipkin(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_ipkin(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_ipkin", 3, &cint1e_ipkin, NPY_COMPLEX, args);
}

int cint1e_ipnuc(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_ipnuc(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_ipnuc", 3, &cint1e_ipnuc, NPY_COMPLEX, args);
}

int cint1e_iprinv(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_iprinv(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_iprinv", 3, &cint1e_iprinv, NPY_COMPLEX, args);
}

int cint1e_ipspnucsp(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_ipspnucsp(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_ipspnucsp", 3, &cint1e_ipspnucsp, NPY_COMPLEX, args);
}

int cint1e_ipsprinvsp(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_ipsprinvsp(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_ipsprinvsp", 3, &cint1e_ipsprinvsp, NPY_COMPLEX, args);
}

int cint1e_sr(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_sr(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_sr", 1, &cint1e_sr, NPY_COMPLEX, args);
}

int cint1e_govlp(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_govlp(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_govlp", 3, &cint1e_govlp, NPY_COMPLEX, args);
}

int cint1e_gnuc(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_gnuc(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_gnuc", 3, &cint1e_gnuc, NPY_COMPLEX, args);
}

int cint1e_cg_sa10sa01(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_cg_sa10sa01(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_cg_sa10sa01", 9, &cint1e_cg_sa10sa01, NPY_COMPLEX, args);
}

int cint1e_cg_sa10sp(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_cg_sa10sp(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_cg_sa10sp", 3, &cint1e_cg_sa10sp, NPY_COMPLEX, args);
}

int cint1e_cg_sa10nucsp(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_cg_sa10nucsp(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_cg_sa10nucsp", 3, &cint1e_cg_sa10nucsp, NPY_COMPLEX, args);
}

int cint1e_giao_sa10sa01(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_giao_sa10sa01(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_giao_sa10sa01", 9, &cint1e_giao_sa10sa01, NPY_COMPLEX, args);
}

int cint1e_giao_sa10sp(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_giao_sa10sp(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_giao_sa10sp", 3, &cint1e_giao_sa10sp, NPY_COMPLEX, args);
}

int cint1e_giao_sa10nucsp(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_giao_sa10nucsp(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_giao_sa10nucsp", 3, &cint1e_giao_sa10nucsp, NPY_COMPLEX, args);
}

int cint1e_sa01sp(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_sa01sp(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_sa01sp", 3, &cint1e_sa01sp, NPY_COMPLEX, args);
}

int cint1e_spgsp(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_spgsp(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_spgsp", 3, &cint1e_spgsp, NPY_COMPLEX, args);
}

int cint1e_spgnucsp(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_spgnucsp(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_spgnucsp", 3, &cint1e_spgnucsp, NPY_COMPLEX, args);
}

int cint1e_spgsa01(double *, int *, int *, int, int *, int, double *);
static PyObject *pycint_cint1e_spgsa01(PyObject *self, PyObject *args)
{
        return pycint_cint1e_common("OOOO:cint1e_spgsa01", 9, &cint1e_spgsa01, NPY_COMPLEX, args);
}

void nr_vhf_o3(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_nr_vhf_o3(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:nr_vhf_o3", 1, &nr_vhf_o3, NPY_DOUBLE, args);
}

void nr_vhf_direct_o3(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_nr_vhf_direct_o3(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:nr_vhf_direct_o3", 1, &nr_vhf_direct_o3, NPY_DOUBLE, args);
}

void nr_vhf_grad_o0(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_nr_vhf_grad_o0(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:nr_vhf_grad_o0", 3, &nr_vhf_grad_o0, NPY_DOUBLE, args);
}

void nr_vhf_grad_o1(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_nr_vhf_grad_o1(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:nr_vhf_grad_o1", 3, &nr_vhf_grad_o1, NPY_DOUBLE, args);
}

void nr_vhf_igiao_o2(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_nr_vhf_igiao_o2(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:nr_vhf_igiao_o2", 3, &nr_vhf_igiao_o2, NPY_DOUBLE, args);
}

void rkb_vhf_coul_o3(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_rkb_vhf_coul_o3(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:rkb_vhf_coul_o3", 1, &rkb_vhf_coul_o3, NPY_COMPLEX, args);
}

void rkb_vhf_ll_o3(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_rkb_vhf_ll_o3(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:rkb_vhf_ll_o3", 1, &rkb_vhf_ll_o3, NPY_COMPLEX, args);
}

void rkb_vhf_sl_o3(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_rkb_vhf_sl_o3(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:rkb_vhf_sl_o3", 1, &rkb_vhf_sl_o3, NPY_COMPLEX, args);
}

void rkb_vhf_coul_direct_o3(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_rkb_vhf_coul_direct_o3(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:rkb_vhf_coul_direct_o3", 1, &rkb_vhf_coul_direct_o3, NPY_COMPLEX, args);
}

void rkb_vhf_ll_direct_o3(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_rkb_vhf_ll_direct_o3(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:rkb_vhf_ll_direct_o3", 1, &rkb_vhf_ll_direct_o3, NPY_COMPLEX, args);
}

void rkb_vhf_sl_direct_o3(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_rkb_vhf_sl_direct_o3(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:rkb_vhf_sl_direct_o3", 1, &rkb_vhf_sl_direct_o3, NPY_COMPLEX, args);
}

void rkb_vhf_coul_grad_o1(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_rkb_vhf_coul_grad_o1(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:rkb_vhf_coul_grad_o1", 3, &rkb_vhf_coul_grad_o1, NPY_COMPLEX, args);
}

void rkb_vhf_coul_grad_ll_o1(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_rkb_vhf_coul_grad_ll_o1(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:rkb_vhf_coul_grad_ll_o1", 3, &rkb_vhf_coul_grad_ll_o1, NPY_COMPLEX, args);
}

void rkb_vhf_coul_grad_ls2l_o1(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_rkb_vhf_coul_grad_ls2l_o1(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:rkb_vhf_coul_grad_ls2l_o1", 3, &rkb_vhf_coul_grad_ls2l_o1, NPY_COMPLEX, args);
}

void rkb_vhf_coul_grad_l2sl_o1(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_rkb_vhf_coul_grad_l2sl_o1(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:rkb_vhf_coul_grad_l2sl_o1", 3, &rkb_vhf_coul_grad_l2sl_o1, NPY_COMPLEX, args);
}

void rkb_vhf_coul_grad_xss_o1(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_rkb_vhf_coul_grad_xss_o1(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:rkb_vhf_coul_grad_xss_o1", 3, &rkb_vhf_coul_grad_xss_o1, NPY_COMPLEX, args);
}

void rmb4cg_vhf_coul(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_rmb4cg_vhf_coul(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:rmb4cg_vhf_coul", 3, &rmb4cg_vhf_coul, NPY_COMPLEX, args);
}

void rmb4giao_vhf_coul(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_rmb4giao_vhf_coul(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:rmb4giao_vhf_coul", 3, &rmb4giao_vhf_coul, NPY_COMPLEX, args);
}

void rkb_giao_vhf_coul(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_rkb_giao_vhf_coul(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:rkb_giao_vhf_coul", 3, &rkb_giao_vhf_coul, NPY_COMPLEX, args);
}

void dkb_vhf_coul(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_dkb_vhf_coul(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:dkb_vhf_coul", 1, &dkb_vhf_coul, NPY_COMPLEX, args);
}

void dkb_vhf_coul_direct(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_dkb_vhf_coul_direct(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:dkb_vhf_coul_direct", 1, &dkb_vhf_coul_direct, NPY_COMPLEX, args);
}

void dkb_vhf_coul_o02(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_dkb_vhf_coul_o02(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:dkb_vhf_coul_o02", 1, &dkb_vhf_coul_o02, NPY_COMPLEX, args);
}

void rkb_vhf_gaunt(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_rkb_vhf_gaunt(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:rkb_vhf_gaunt", 1, &rkb_vhf_gaunt, NPY_COMPLEX, args);
}

void rkb_vhf_gaunt_direct(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_rkb_vhf_gaunt_direct(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:rkb_vhf_gaunt_direct", 1, &rkb_vhf_gaunt_direct, NPY_COMPLEX, args);
}

void rmb4cg_vhf_gaunt(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_rmb4cg_vhf_gaunt(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:rmb4cg_vhf_gaunt", 3, &rmb4cg_vhf_gaunt, NPY_COMPLEX, args);
}

void rmb4giao_vhf_gaunt(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_rmb4giao_vhf_gaunt(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:rmb4giao_vhf_gaunt", 3, &rmb4giao_vhf_gaunt, NPY_COMPLEX, args);
}

void rkb_giao_vhf_gaunt(double *, double *, double *, int, int, int,
int *, int, int *, int, double *);
static PyObject *pycint_rkb_giao_vhf_gaunt(PyObject *self, PyObject *args)
{
        return pycint_vhf_common("OOOO:rkb_giao_vhf_gaunt", 3, &rkb_giao_vhf_gaunt, NPY_COMPLEX, args);
}

static PyMethodDef pycintMethods[] = {
{"nr_vhf_o3", pycint_nr_vhf_o3, METH_VARARGS, doc_vhf_common},
{"nr_vhf_direct_o3", pycint_nr_vhf_direct_o3, METH_VARARGS, doc_vhf_common},
{"nr_vhf_grad_o0", pycint_nr_vhf_grad_o0, METH_VARARGS, doc_vhf_common},
{"nr_vhf_grad_o1", pycint_nr_vhf_grad_o1, METH_VARARGS, doc_vhf_common},
{"nr_vhf_igiao_o2", pycint_nr_vhf_igiao_o2, METH_VARARGS, doc_vhf_common},
{"rkb_vhf_coul_o3", pycint_rkb_vhf_coul_o3, METH_VARARGS, doc_vhf_common},
{"rkb_vhf_ll_o3", pycint_rkb_vhf_ll_o3, METH_VARARGS, doc_vhf_common},
{"rkb_vhf_sl_o3", pycint_rkb_vhf_sl_o3, METH_VARARGS, doc_vhf_common},
{"rkb_vhf_coul_direct_o3", pycint_rkb_vhf_coul_direct_o3, METH_VARARGS, doc_vhf_common},
{"rkb_vhf_ll_direct_o3", pycint_rkb_vhf_ll_direct_o3, METH_VARARGS, doc_vhf_common},
{"rkb_vhf_sl_direct_o3", pycint_rkb_vhf_sl_direct_o3, METH_VARARGS, doc_vhf_common},
{"rkb_vhf_coul_grad_o1", pycint_rkb_vhf_coul_grad_o1, METH_VARARGS, doc_vhf_common},
{"rkb_vhf_coul_grad_ll_o1", pycint_rkb_vhf_coul_grad_ll_o1, METH_VARARGS, doc_vhf_common},
{"rkb_vhf_coul_grad_ls2l_o1", pycint_rkb_vhf_coul_grad_ls2l_o1, METH_VARARGS, doc_vhf_common},
{"rkb_vhf_coul_grad_l2sl_o1", pycint_rkb_vhf_coul_grad_l2sl_o1, METH_VARARGS, doc_vhf_common},
{"rkb_vhf_coul_grad_xss_o1", pycint_rkb_vhf_coul_grad_xss_o1, METH_VARARGS, doc_vhf_common},
{"rmb4cg_vhf_coul", pycint_rmb4cg_vhf_coul, METH_VARARGS, doc_vhf_common},
{"rmb4giao_vhf_coul", pycint_rmb4giao_vhf_coul, METH_VARARGS, doc_vhf_common},
{"rkb_giao_vhf_coul", pycint_rkb_giao_vhf_coul, METH_VARARGS, doc_vhf_common},
{"dkb_vhf_coul", pycint_dkb_vhf_coul, METH_VARARGS, doc_vhf_common},
{"dkb_vhf_coul_direct", pycint_dkb_vhf_coul_direct, METH_VARARGS, doc_vhf_common},
{"dkb_vhf_coul_o02", pycint_dkb_vhf_coul_o02, METH_VARARGS, doc_vhf_common},
{"rkb_vhf_gaunt", pycint_rkb_vhf_gaunt, METH_VARARGS, doc_vhf_common},
{"rkb_vhf_gaunt_direct", pycint_rkb_vhf_gaunt_direct, METH_VARARGS, doc_vhf_common},
{"rmb4cg_vhf_gaunt", pycint_rmb4cg_vhf_gaunt, METH_VARARGS, doc_vhf_common},
{"rmb4giao_vhf_gaunt", pycint_rmb4giao_vhf_gaunt, METH_VARARGS, doc_vhf_common},
{"rkb_giao_vhf_gaunt", pycint_rkb_giao_vhf_gaunt, METH_VARARGS, doc_vhf_common},
{"cint1e_ovlp_sph", pycint_cint1e_ovlp_sph, METH_VARARGS, doc_cint1e_common},
{"cint1e_kin_sph", pycint_cint1e_kin_sph, METH_VARARGS, doc_cint1e_common},
{"cint1e_nuc_sph", pycint_cint1e_nuc_sph, METH_VARARGS, doc_cint1e_common},
{"cint1e_giao_irjxp_sph", pycint_cint1e_giao_irjxp_sph, METH_VARARGS, doc_cint1e_common},
{"cint1e_cg_irxp_sph", pycint_cint1e_cg_irxp_sph, METH_VARARGS, doc_cint1e_common},
{"cint1e_ia01p_sph", pycint_cint1e_ia01p_sph, METH_VARARGS, doc_cint1e_common},
{"cint1e_giao_a11part_sph", pycint_cint1e_giao_a11part_sph, METH_VARARGS, doc_cint1e_common},
{"cint1e_cg_a11part_sph", pycint_cint1e_cg_a11part_sph, METH_VARARGS, doc_cint1e_common},
{"cint1e_igovlp_sph", pycint_cint1e_igovlp_sph, METH_VARARGS, doc_cint1e_common},
{"cint1e_igkin_sph", pycint_cint1e_igkin_sph, METH_VARARGS, doc_cint1e_common},
{"cint1e_ignuc_sph", pycint_cint1e_ignuc_sph, METH_VARARGS, doc_cint1e_common},
{"cint1e_a01gp_sph", pycint_cint1e_a01gp_sph, METH_VARARGS, doc_cint1e_common},
{"cint1e_ipovlp_sph", pycint_cint1e_ipovlp_sph, METH_VARARGS, doc_cint1e_common},
{"cint1e_ipkin_sph", pycint_cint1e_ipkin_sph, METH_VARARGS, doc_cint1e_common},
{"cint1e_ipnuc_sph", pycint_cint1e_ipnuc_sph, METH_VARARGS, doc_cint1e_common},
{"cint1e_iprinv_sph", pycint_cint1e_iprinv_sph, METH_VARARGS, doc_cint1e_common},
{"cint1e_ovlp", pycint_cint1e_ovlp, METH_VARARGS, doc_cint1e_common},
{"cint1e_spsp", pycint_cint1e_spsp, METH_VARARGS, doc_cint1e_common},
{"cint1e_nuc", pycint_cint1e_nuc, METH_VARARGS, doc_cint1e_common},
{"cint1e_spnucsp", pycint_cint1e_spnucsp, METH_VARARGS, doc_cint1e_common},
{"cint1e_srnucsr", pycint_cint1e_srnucsr, METH_VARARGS, doc_cint1e_common},
{"cint1e_sp", pycint_cint1e_sp, METH_VARARGS, doc_cint1e_common},
{"cint1e_srsr", pycint_cint1e_srsr, METH_VARARGS, doc_cint1e_common},
{"cint1e_srsp", pycint_cint1e_srsp, METH_VARARGS, doc_cint1e_common},
{"cint1e_spspsp", pycint_cint1e_spspsp, METH_VARARGS, doc_cint1e_common},
{"cint1e_spnuc", pycint_cint1e_spnuc, METH_VARARGS, doc_cint1e_common},
{"cint1e_ipovlp", pycint_cint1e_ipovlp, METH_VARARGS, doc_cint1e_common},
{"cint1e_ipkin", pycint_cint1e_ipkin, METH_VARARGS, doc_cint1e_common},
{"cint1e_ipnuc", pycint_cint1e_ipnuc, METH_VARARGS, doc_cint1e_common},
{"cint1e_iprinv", pycint_cint1e_iprinv, METH_VARARGS, doc_cint1e_common},
{"cint1e_ipspnucsp", pycint_cint1e_ipspnucsp, METH_VARARGS, doc_cint1e_common},
{"cint1e_ipsprinvsp", pycint_cint1e_ipsprinvsp, METH_VARARGS, doc_cint1e_common},
{"cint1e_sr", pycint_cint1e_sr, METH_VARARGS, doc_cint1e_common},
{"cint1e_govlp", pycint_cint1e_govlp, METH_VARARGS, doc_cint1e_common},
{"cint1e_gnuc", pycint_cint1e_gnuc, METH_VARARGS, doc_cint1e_common},
{"cint1e_cg_sa10sa01", pycint_cint1e_cg_sa10sa01, METH_VARARGS, doc_cint1e_common},
{"cint1e_cg_sa10sp", pycint_cint1e_cg_sa10sp, METH_VARARGS, doc_cint1e_common},
{"cint1e_cg_sa10nucsp", pycint_cint1e_cg_sa10nucsp, METH_VARARGS, doc_cint1e_common},
{"cint1e_giao_sa10sa01", pycint_cint1e_giao_sa10sa01, METH_VARARGS, doc_cint1e_common},
{"cint1e_giao_sa10sp", pycint_cint1e_giao_sa10sp, METH_VARARGS, doc_cint1e_common},
{"cint1e_giao_sa10nucsp", pycint_cint1e_giao_sa10nucsp, METH_VARARGS, doc_cint1e_common},
{"cint1e_sa01sp", pycint_cint1e_sa01sp, METH_VARARGS, doc_cint1e_common},
{"cint1e_spgsp", pycint_cint1e_spgsp, METH_VARARGS, doc_cint1e_common},
{"cint1e_spgnucsp", pycint_cint1e_spgnucsp, METH_VARARGS, doc_cint1e_common},
{"cint1e_spgsa01", pycint_cint1e_spgsa01, METH_VARARGS, doc_cint1e_common},
{NULL, NULL, 0, NULL}
};
