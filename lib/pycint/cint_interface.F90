!============================================================
!
! File:         cint_interface.F90
! Author:       Qiming Sun <qiming.sqm@gmail.com>
! Last change:
! Version:
! Description:  interface of contracted GTO integrals
!
!============================================================

module cint_interface
  interface

integer function CINTcgto_cart(bas_id, bas)
  integer,intent(in)    ::  bas_id
  integer,intent(in)    ::  bas(*)
end function CINTcgto_cart

integer function CINTcgto_spheric(bas_id, bas)
  integer,intent(in)    ::  bas_id
  integer,intent(in)    ::  bas(*)
end function CINTcgto_spheric

integer function CINTcgto_spinor(bas_id, bas)
  integer,intent(in)    ::  bas_id
  integer,intent(in)    ::  bas(*)
end function CINTcgto_spinor

integer function CINTlen_spinor(bas_id, bas)
  integer,intent(in)    ::  bas_id
  integer,intent(in)    ::  bas(*)
end function CINTlen_spinor

integer function CINTtot_pgto_spheric(bas, nbas)
  integer,intent(in)    ::  nbas
  integer,intent(in)    ::  bas(*)
end function CINTtot_pgto_spheric

integer function CINTtot_pgto_spinor(bas, nbas)
  integer,intent(in)    ::  nbas
  integer,intent(in)    ::  bas(*)
end function CINTtot_pgto_spinor

integer function CINTtot_cgto_cart(bas, nbas)
  integer,intent(in)    ::  nbas
  integer,intent(in)    ::  bas(*)
end function CINTtot_cgto_cart

integer function CINTtot_cgto_spheric(bas, nbas)
  integer,intent(in)    ::  nbas
  integer,intent(in)    ::  bas(*)
end function CINTtot_cgto_spheric

integer function CINTtot_cgto_spinor(bas, nbas)
  integer,intent(in)    ::  nbas
  integer,intent(in)    ::  bas(*)
end function CINTtot_cgto_spinor

subroutine CINTshells_cart_offset(ao_loc, bas, nbas)
  integer,intent(out)   ::  ao_loc(*)
  integer,intent(in)    ::  nbas
  integer,intent(in)    ::  bas(*)
end subroutine CINTshells_cart_offset

subroutine CINTshells_spheric_offset(ao_loc, bas, nbas)
  integer,intent(out)   ::  ao_loc(*)
  integer,intent(in)    ::  nbas
  integer,intent(in)    ::  bas(*)
end subroutine CINTshells_spheric_offset

subroutine CINTshells_spinor_offset(ao_loc, bas, nbas)
  integer,intent(out)   ::  ao_loc(*)
  integer,intent(in)    ::  nbas
  integer,intent(in)    ::  bas(*)
end subroutine CINTshells_spinor_offset
  end interface
end module cint_interface
