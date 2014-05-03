!============================================================
!
! File:         cint_const_mod.F90
! Author:       Qiming Sun <qiming.sqm@gmail.com>
! Last change:
! Version:
! Description:  common variables of contracted GTO integrals
!
!============================================================


module cint_const_mod

  ! global parameters in env
  integer,parameter     ::  PTR_LIGHT_SPEED = 1
  integer,parameter     ::  PTR_COMMON_ORIG = 2
  integer,parameter     ::  PTR_RINV_ORIG   = 5
  integer,parameter     ::  PTR_AO_GAUGE    = 8
  integer,parameter     ::  PTR_ENV_START   = 20

  ! slots of each atom
  integer,parameter     ::  CHARGE_OF  = 1
  integer,parameter     ::  PTR_COORD  = 2
  integer,parameter     ::  NUC_MOD_OF = 3
  integer,parameter     ::  PTR_MASS   = 4
  integer,parameter     ::  RAD_GRIDS = 5
  integer,parameter     ::  ANG_GRIDS  = 6
  integer,parameter     ::  ATM_SLOTS  = 6

  ! slots of each basis
  integer,parameter     ::  ATOM_OF    = 1
  integer,parameter     ::  ANG_OF     = 2  ! angular momentum
  integer,parameter     ::  NPRIM_OF   = 3  ! num. of primitive GTO
  integer,parameter     ::  NCTR_OF    = 4  ! num. of contrancted GTO
  integer,parameter     ::  KAPPA_OF   = 5
  integer,parameter     ::  PTR_EXP    = 6
  integer,parameter     ::  PTR_COEFF  = 7
  integer,parameter     ::  RESERVE_BASLOT = 8
  integer,parameter     ::  BAS_SLOTS  = 8

  ! POS_e2pauli
  integer,parameter     ::  POS_X = 1
  integer,parameter     ::  POS_Y = 2
  integer,parameter     ::  POS_Z = 3
  integer,parameter     ::  POS_1 = 4

  ! POS_e1pauli_e2pauli
  integer,parameter     ::  POS_X_X = 1 
  integer,parameter     ::  POS_Y_X = 2 
  integer,parameter     ::  POS_Z_X = 3 
  integer,parameter     ::  POS_1_X = 4 
  integer,parameter     ::  POS_X_Y = 5 
  integer,parameter     ::  POS_Y_Y = 6 
  integer,parameter     ::  POS_Z_Y = 7 
  integer,parameter     ::  POS_1_Y = 8 
  integer,parameter     ::  POS_X_Z = 9 
  integer,parameter     ::  POS_Y_Z = 10
  integer,parameter     ::  POS_Z_Z = 11
  integer,parameter     ::  POS_1_Z = 12
  integer,parameter     ::  POS_X_1 = 13
  integer,parameter     ::  POS_Y_1 = 14
  integer,parameter     ::  POS_Z_1 = 15
  integer,parameter     ::  POS_1_1 = 16

  ! No. Rys. roots
  integer,parameter     ::  MXRYSROOTS = 14
end module cint_const_mod
