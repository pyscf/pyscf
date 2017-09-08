module m_param_arr_type

#include "m_define_macro.F90"
  use m_param, only : param_t

  
  implicit none

  type param_arr_t
    type(param_t), allocatable :: a(:)
    integer :: n = -999 
  end type ! param_arr_t
    
end module !m_param_arr_type
