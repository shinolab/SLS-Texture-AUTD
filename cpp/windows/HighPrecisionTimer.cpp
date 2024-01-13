/*
 * @Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
 * @Date: 2022-12-19 16:32:53
 * @LastEditors: Mingxin Zhang
 * @LastEditTime: 2022-12-19 16:58:15
 * Copyright (c) 2022 by Mingxin Zhang, All Rights Reserved. 
 */

#include <iostream>
#include <windows.h>

class HighPrecisionTimer
{
    public:
        float HighPrecisionSleep(float deltaTime);
};

float HighPrecisionTimer::HighPrecisionSleep(float deltaTime)
{
    LARGE_INTEGER freq;
    if (!QueryPerformanceFrequency(&freq))
        return 0;
    LARGE_INTEGER start, end;

    QueryPerformanceCounter(&start);
    QueryPerformanceCounter(&end);
    auto elapsedTime = 0.0;
    while (elapsedTime < deltaTime) {
        QueryPerformanceCounter(&end);
        elapsedTime = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart;
    }
    return elapsedTime;
}

extern "C" {
    HighPrecisionTimer Timer;
    float HighPrecisionSleep(float deltaTime)
    {
        return Timer.HighPrecisionSleep(deltaTime);
    }
}