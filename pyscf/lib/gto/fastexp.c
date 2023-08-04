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

 *
 *      Fast exp(x) computation (with SSE2 optimizations).
 *
 * Copyright (c) 2010, Naoaki Okazaki
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the names of the authors nor the names of its contributors
 *       may be used to endorse or promote products derived from this
 *       software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Modified from   http://www.chokkan.org/blog/archives/352
 *                 http://www.chokkan.org/software/dist/fastexp.c
 */

#include <math.h>

typedef union {
    double d;
    unsigned short s[4];
} ieee754;

static const double LOG2E  =  1.4426950408889634073599;     /* 1/log(2) */
//static const double INFINITY = 1.79769313486231570815E308;
static const double C1 = 6.93145751953125E-1;
static const double C2 = 1.42860682030941723212E-6;

double exp_cephes(double x)
{
        int n;
        double a, xx, px, qx;
        ieee754 u;

        /* n = round(x / log 2) */
        a = LOG2E * x + 0.5;
        n = (int)a;
        n -= (a < 0);

        /* x -= n * log2 */
        px = (double)n;
        x -= px * C1;
        x -= px * C2;
        xx = x * x;

        /* px = x * P(x**2). */
        px = 1.26177193074810590878E-4;
        px *= xx;
        px += 3.02994407707441961300E-2;
        px *= xx;
        px += 9.99999999999999999910E-1;
        px *= x;

        /* Evaluate Q(x**2). */
        qx = 3.00198505138664455042E-6;
        qx *= xx;
        qx += 2.52448340349684104192E-3;
        qx *= xx;
        qx += 2.27265548208155028766E-1;
        qx *= xx;
        qx += 2.00000000000000000009E0;

        /* e**x = 1 + 2x P(x**2)/( Q(x**2) - P(x**2) ) */
        x = px / (qx - px);
        x = 1.0 + 2.0 * x;

        /* Build 2^n in double. */
        u.d = 0;
        n += 1023;
        u.s[3] = (unsigned short)((n << 4) & 0x7FF0);

        return x * u.d;
}

