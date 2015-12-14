#ifndef INTEGRALIMG_H
#define INTEGRALIMG_H

#include "types.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "kernel.cuh"

__global__ void scanRow(float *input, float *output, float *aux, int len);
__global__ void parallelScanAddAux(float *aux, float *output, int len);

int CudaIntegralImage(float* grayImage, float* integralImage, int width, int height);


#endif INTEGRALIMG_H