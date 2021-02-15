/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
  
   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
 
        http://www.apache.org/licenses/LICENSE-2.0
 
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
 */

/*
#include "elapse.h"
*/

#include <sys/time.h>
#ifdef _AIX
void cputime(double *val)
#else
void cputime_(double *val)
#endif
{
  struct timeval time;
  gettimeofday(&time,(struct timezone *)0);
  *val=time.tv_sec+time.tv_usec*1.e-6;
}

#include <unistd.h>
#ifdef _AIX
void getpid(int *val)
#else
void getpid_(int *val)
#endif
{
  *val = getpid();
}
