#ifndef HAAR_H
#define HAAR_H

#include "types.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "kernel.cuh"

__global__ void naiveCudaHaarKernel(float* deviceIntegralImage, int width, int height, int winWidth, int winHeight, float scale, int step);
__device__ __host__ float rectSum(float const* image, int imWidth, int inHeight, int x, int y, int w, int h);

int CudaAllScalesCascade(unsigned char* outputImage, const float* integralImage, int width, int height);
int CudaHaarCascade(unsigned char* outputImage, const float* integralImage, int width, int height);

#endif HAAR_H