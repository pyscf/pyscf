! Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
!
! Licensed under the Apache License, Version 2.0 (the "License");
!  you may not use this file except in compliance with the License.
!  You may obtain a copy of the License at
!
!      http://www.apache.org/licenses/LICENSE-2.0
!
!  Unless required by applicable law or agreed to in writing, software
!  distributed under the License is distributed on an "AS IS" BASIS,
!  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
!  See the License for the specific language governing permissions and
!  limitations under the License.

#define _dealloc(a) if(allocated(a)) deallocate(a)

#ifdef DEBUG
#define _G write(6,'(a,1x,a,1x,i8)') 'GREETS from', __FILE__, __LINE__
#else
#define _G 
#endif

#ifdef TIMING
#define _T write(6,'(a,1x,a18,1x,i9,1x,a)')'TIMING ', get_cdatetime(), __LINE__, __FILE__
#define _TM(m) write(6,'(a,1x,a18,1x,a,i9,1x,a)')'TIMING ', get_cdatetime(), m, __LINE__, __FILE__
#else
#define _T
#define _TM(m)
#endif

#define _assert(a) if(.not. allocated(a)) call die('a '//__FILE__, __LINE__)
#define _add_size_alloc(n,a)if(allocated(a))n=n+size(a)
#define _size_alloc(a,n)n=0;if(allocated(a))n=size(a)
#define _add_size_array(n,a) n = n + size(a)
#define _die(m) call die(m//' in '//__FILE__, __LINE__)
#define _warn(m) call warn(m//' in '//__FILE__, __LINE__)
#define _print(m) write(6,*) m, 'm'
#define _mem_note call log_memory_note(__FILE__, 0, __LINE__)
#define _MN call log_memory_note(__FILE__, 0, __LINE__)
#ifdef MPROF
#define _MP(l) call mprof(l, __LINE__);
#else
#define _MP(l)
#endif
#define _mem_note_iter(iteration) call log_memory_note(__FILE__, 0, __LINE__, iteration)
#define _zero(a) if(allocated(a)) a=0
#define _pack_indx(i,j) j*(j-1)/2+i
#define _pack_size(n) n*(n+1)/2
#define _t1 call cputime(t1)
#define _t2(t)call cputime(t2);t=t+(t2-t1);call cputime(t1);
#define _conc(a,b) trim(a)//', '//trim(b)
#define _realloc1(a,l,u) if(allocated(a))then;if(any(lbound(a)/=l) .or. any(ubound(a)/=u)) deallocate(a);endif 
#define _realloc(a,n) if(allocated(a)) deallocate(a); allocate(a(n))
#define _realloc2(a,n,m) if(allocated(a)) deallocate(a); allocate(a(n,m))
#define _dealloc_u(a,u)if(allocated(a))then;if(any(ubound(a)/=u))deallocate(a);endif;
#define _dealloc_lu(a,l,u)if(allocated(a))then;\
if(any(ubound(a)/=u).or.any(lbound(a)/=l))deallocate(a);\
endif
#define _dea_d(a,d)if(allocated(a))then;if(any(ubound(a)/=d(1:1,2)).or.any(lbound(a)/=d(1:1,1)))deallocate(a);endif;
#define _get_d(a) reshape([lbound(a), ubound(a)], (/size(shape(a)), 2/))
#define _INIT_ERROR(error)  if (present(error)) then ; error = ERROR_NONE ; endif
#define _where write(6,'(a,1x,(" at "),i9)')__FILE__, __LINE__

#define _size_mem(a) size(a)+2*size(shape(a))
#define _incr_sf(p,a,f)p=p+1;f=p+size(a)-1+2*size(shape(a));
#define _poke_lub(p,a,mem)mem(p:p+2*size(shape(a))-1)=[lbound(a),ubound(a)];p=p+2*size(shape(a));
#define _poke_dat(p,f,a,mem)mem(p:f)=reshape(a,[f-p+1]);p=f;
#define _get_lbubnn(p,d,m,lb,ub,nn)lb(1:d)=int(m(p:p+d-1));ub(1:d)=int(m(p+d:p+d+d-1));nn(1:d)=ub(1:d)-lb(1:d)+1;
#define _incr_sf_nn(p,d,nn,f)p=f+1;f=p+product(nn(1:d))-1;
#define _pause write(0,*) 'pause at: '//__FILE__,__LINE__; read(*,*);
#define _warn_dev call warn_dev(__FILE__, __LINE__);
