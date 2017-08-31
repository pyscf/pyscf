"""
!ccc Copyright (C) 2004-2009: Leslie Greengard and June-Yub Lee 
!ccc Contact: greengard@cims.nyu.edu
!ccc 
!ccc This software is being released under a FreeBSD license (see below). 
!ccc
!cc
!cc Copyright (c) 2009-2013, Leslie Greengard, June-Yub Lee and Zydrunas Gimbutas
!cc All rights reserved.
!
!cc Redistribution and use in source and binary forms, with or without
!cc modification, are permitted provided that the following conditions are met: 
!
!cc 1. Redistributions of source code must retain the above copyright notice, this
!cc    list of conditions and the following disclaimer. 
!cc 2. Redistributions in binary form must reproduce the above copyright notice,
!cc    this list of conditions and the following disclaimer in the documentation
!cc    and/or other materials provided with the distribution. 
!
!cc THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
!cc ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
!cc WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
!cc DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
!cc ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
!cc (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
!cc LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
!cc ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
!cc (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
!cc SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
!
!cc The views and conclusions contained in the software and documentation are those
!cc of the authors and should not be interpreted as representing official policies, 
!cc either expressed or implied, of the FreeBSD Project.
!

module m_next235

  
  contains 

!
!************************************************************************
!
function next235(base)
  implicit none
  !! external
  real(8), intent(in) :: base
  integer :: next235  
  !! internal
  integer :: numdiv
  
!c ----------------------------------------------------------------------
!c     integer function next235 returns a multiple of 2, 3, and 5
!c
!c     next235 = 2^p 3^q 5^r >= base  where p>=1, q>=0, r>=0
!************************************************************************
      next235 = 2 * int(base/2d0+.9999d0)
      if (next235.le.0) next235 = 2

100   numdiv = next235
      do while (numdiv/2*2 .eq. numdiv)
         numdiv = numdiv /2
      enddo
      do while (numdiv/3*3 .eq. numdiv)
         numdiv = numdiv /3
      enddo
      do while (numdiv/5*5 .eq. numdiv)
         numdiv = numdiv /5
      enddo
      if (numdiv .eq. 1) return
      next235 = next235 + 2
      goto 100
end function ! next235


end module !m_next235
"""

def next235(base):
  assert(type(base)==float)
  next235 = 2 * int(base/2.0+.9999)
  if (next235<=0) : next235 = 2
  
  return(int(base))
  
