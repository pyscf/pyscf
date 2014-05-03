/*
 * File: misc.c
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 *
 */

#include <sys/time.h>
#include <unistd.h>

// first place has bit 1. e.g. first_bit1(5) = 2
int first_bit1(int n)
{
        int acc = 0;
        while (n) {
                n = n >> 1;
                acc++;
        }
        return acc-1;
}


double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time, NULL)) {
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * 1e-6;
}
