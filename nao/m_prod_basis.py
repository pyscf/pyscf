from __future__ import division
import numpy as np
  

class prod_basis_c():
  """
    character(100) :: prod_basis_type = ""
    integer :: check_basis            =-999  ! usual test to perform or skip (overlaps and dipoles comparison)
    integer :: check_basis_rspace     =-999  ! usual test to perform or skip (overlaps and dipoles comparison) / moments 
    integer :: check_basis_bloch      =-999  ! recompute k-dependent overlaps out of real-space overlaps.
    integer :: check_basis_write_report =-999! whether to write a file with a report on usual test of dominant products
    integer :: cross_check_global_vertex = -999 ! computes the global vertex coefficients and compares
    integer :: check_zcmmm               = -999 ! check matmuls with the conversion matrix (CM) ac<->dp (CM is stored block-sparse)
    integer :: check_dfmcm_lap           = -999 ! check a list-based matmul with the conversion matrix (CM) 
    integer :: check_zspcm_lap           = -999 ! check a list-based matmul of symmetric,packed matrix with the conversion matrix (CM) 
    integer :: check_exmtrx_vt1_bt2      = -999 ! check an implementation of exchange matrix with atom-centered kernel and dominant product vertex
    
    integer :: stop_check_basis_bloch   =-999! whether to stop after an eventual checking of bloch's overlaps
    integer :: do_shifting_test =-999! additional test to perform or skip 
    integer :: metric_type      =-999! 1-- Cartesian metric, 2 -- Coulomb metric
    integer :: normalize_dp     =-999! whether we want to normalize dominant functions after determining them
    real(8) :: eigmin_local     =-999! eigenvalue threshold to generate local functions
    real(8) :: eigmin_bilocal   =-999! eigenvalue threshold to generate bilocal functions
    integer :: optimize_centers =-999! whether we try to optimize centers of bilocal functions
    integer :: jcutoff          =-999! angular momentum cutoff (desired) for expansion of bilocal dominant products
    integer :: gl_ord_bilocal   =-999! parameters for numerical integration in construction of bilocal dominant products
    integer :: gl_pcs_bilocal   =-999! parameters for numerical integration in construction of bilocal dominant products
    real(8) :: ac_rcut          =-999! radius of a sphere to determine which (atom centered) functions contribute or belong to a given center.
    character(99) :: ac_method  = "" ! Method to choose the contributing centers: LENS or (SPHERE).
    character(99) :: cc_method  = "" ! Method to compute conversion coefficients: (DSYEV), DSYEVR, DSYTR
    character(99) :: cc_omp_para= "" ! Parallelization approach for computation of conversion coefficients PAIRS, ATOMS, (AUTO)
    real(8) :: cc_inv_eigmin = -999  ! Threshold for dropping small eigenvalues in comput. of convers. coeffs.
    integer :: check_hkernel_blocks =-999! whether we will (cross) check blocks of Hartree kernel in mixed basis construction
    integer :: check_hkernel_dp_pb  =-999! generates pb1 = (dp->pb) and pb2, computes kernels and compares the kernels
    integer :: reexpr_further =-999! tells whether the further computation should be done with quantities that are tensors in dp basis set,
                              ! but computed indirectly, by converting from a (smaller) reexpressing basis set.
    integer :: check_hkernel_reexpr =-999 ! test reexpression of dominant products by comparing hkernel_reexpr with hkernel_orig (in dp basis)
    integer :: report_pair2clist  =-999! flag to cause a report on pair2clist structure (can be a lengthy file)
    integer :: report_ls_blocks   =-999! flag to cause a report on ls_blocks structure (can be a lengthy file)
    integer :: bulk_cross_check_hkernel =-999! flag to cause a rather extensive cross check against 
                                        ! a finite-size calculation which includes all super cell
    integer :: report_domiprod       =-999! report main metrics of the initialized type domiprod_t 
    integer :: report_prod_basis_mix =-999! report prod basis with reexpressing coefficients
    integer :: report_prod_basis_dp  =-999! report prod basis when it is still dominant products only
    integer :: report_prod_basis     =-999! report prod basis just before testing
    integer :: gen_aug_lmult         =-999! generate augmented l-multipletts 
    character(100) :: bilocal_type   = "" ! ATOM or MULT
    character(100) :: bilocal_center = "" ! algorithm to determine centers for expansion of bilocal product
    real(8)        :: bilocal_center_pow = -999 ! weight = (coeff * sqrt(<r^2>)**pow)^-1
    character(100) :: bilocal_center_coeff = "" ! 2*L+1, for example    
    character(100) :: local_type     = "" ! ATOM or SHELL
    real(8)        :: core_split_ratio = -999 ! for decision whether an orbital belongs to core
    integer        :: stop_after_dp = -999
    integer        :: pb_funct_xsf = -999 ! whether to plot product functions to xsf files...
    character(100) :: biloc_interm_storage = "" ! type of intermediate storage for generation of bilocal dominant products
    character(100) :: mixed_interm_storage = "" ! type of intermediate storage for generation of reexpression coefficients
  """

  def __init__(self, sv, input_params_pb = {}):
    
    if sv.wfsx.gamma: check_basis_bloch = True
    else: check_basis_bloch = False
    jmx = np.max(sv.mu_sp2j)

    # setup the default value for the inputs
    self.pb_params = {'prod_basis_type': "DOMIPROD",
                     'check_basis': True,
                     'check_basis_rspace': True,
                     'check_basis_bloch': check_basis_bloch,
                     'check_basis_write_report': False,
                     'cross_check_global_vertex': False,
                     'check_zcmmm': False,
                     'check_dfmcm_lap': False,
                     'check_zspcm_lap': False,
                     'check_exmtrx_vt1_bt2': False,
                      
                     'stop_check_basis_bloch': False,
                     'do_shifting_test': False,
                     'metric_type': 2,
                     'normalize_dp': False,
                     'eigmin_local': 1e-4,
                     'eigmin_bilocal': 1e-5,
                      
                     'jcutoff': max(5+jmx, 2*jmx),
                     'optimize_centers': 0,
                     'gl_ord_bilocal': 96,
                     'gl_pcs_bilocal': 1,
                      
                     'ac_rcut': np.max(sv.mu_sp2rcut),
                     'ac_method': "SPHERE",
                     'cc_method': "DSYEV",
                     'cc_omp_para': "AUTO",
                     'cc_inv_eigmin': 1E-9,
                     
                     'check_hkernel_blocks': False,
                     'check_hkernel_dp_pb': False,
                     'reexpr_further': False,
                     'check_hkernel_reexpr': False,
                     'report_pair2clist': False,
                     'report_ls_blocks': False,
                     'bulk_cross_check_hkernel': False,
                      
                     'report_domiprod': True,
                     'report_prod_basis_mix': False,
                     'report_prod_basis_dp': False,
                     'report_prod_basis': True,
                     'gen_aug_lmult': False,
                     
                     'bilocal_type': "ATOM",
                     'bilocal_center': "POW&COEFF",
                     'bilocal_center_pow': 1.0,
                     'bilocal_center_coeff': "2*L+1",
                     'local_type': "ATOM",
                     'core_split_ratio': 10.0,
                     'stop_after_dp': False,
                     'pb_funct_xsf': False,
                     'biloc_interm_storage': "RD",
                     'mixed_interm_storage': "RD"}

    for key, val in input_params_pb.items():
      if key not in self.pb_params.keys():
        raise ValueError(key + " not a pb_param!")
      else:
        if not isinstance(val, type(self.pb_params[key])):
          raise ValueError("Wrong type for " + key + " parameter in pb_param!")
        else:
          self.pb_params[key] = val
