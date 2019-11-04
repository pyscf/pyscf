/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <multithreading.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
//Create thread
CUTThread cutStartThread(CUT_THREADROUTINE func, void *data)
{
    return CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)func, data, 0, NULL);
}

//Wait for thread to finish
void cutEndThread(CUTThread thread)
{
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
}

//Destroy thread
void cutDestroyThread(CUTThread thread)
{
    TerminateThread(thread, 0);
    CloseHandle(thread);
}

//Wait for multiple threads
void cutWaitForThreads(const CUTThread *threads, int num)
{
    WaitForMultipleObjects(num, threads, true, INFINITE);

    for (int i = 0; i < num; i++)
    {
        CloseHandle(threads[i]);
    }
}

#else
//Create thread
CUTThread cutStartThread(CUT_THREADROUTINE func, void *data)
{
    pthread_t thread;
    pthread_create(&thread, NULL, func, data);
    return thread;
}

//Wait for thread to finish
void cutEndThread(CUTThread thread)
{
    pthread_join(thread, NULL);
}

//Destroy thread
void cutDestroyThread(CUTThread thread)
{
    pthread_cancel(thread);
}

//Wait for multiple threads
void cutWaitForThreads(const CUTThread *threads, int num)
{
    for (int i = 0; i < num; i++)
    {
        cutEndThread(threads[i]);
    }
}

#endif
