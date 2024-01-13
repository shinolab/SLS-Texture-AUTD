/*
 * @Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
 * @Date: 2022-12-19 16:32:53
 * @LastEditors: Mingxin Zhang
 * @LastEditTime: 2023-01-12 13:29:51
 * Copyright (c) 2022 by Mingxin Zhang, All Rights Reserved. 
 */

#include <iostream>
#include <sys/time.h>

class HighPrecisionTimer
{
    public:
        float HighPrecisionSleep(float deltaTime);
};

float HighPrecisionTimer::HighPrecisionSleep(float deltaTime)
{
    timeval start, end;
    double elapsedTime = 0.0;  // unit is us
    gettimeofday(&start, NULL);
    while (elapsedTime < deltaTime * 1e6) {
        gettimeofday(&end, NULL);
        elapsedTime = (double)(1000 * 1000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec));
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