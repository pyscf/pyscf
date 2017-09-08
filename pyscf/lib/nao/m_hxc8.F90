module m_hxc8

  use m_block_crs8, only : block_crs8_t
  use m_pb_coul_aux, only : pb_coul_aux_t
  
  implicit none

  !! The type will contain the fields for 
  type hxc8_t
    real(4), allocatable :: hxc_pack(:) ! to hold the kernel in packed form
    type(block_crs8_t)   :: bcrs   ! to hold the kernel of overlapping functions
    type(pb_coul_aux_t)  :: ca      ! auxiliary to compute the Hartree kernel
  end type !hxc8_t

end module !m_hxc8
