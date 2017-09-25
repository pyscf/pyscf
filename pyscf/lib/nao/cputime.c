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
