#ifndef KERNEL_H
#define KERNEL_H

#include "types.h"


#define TILE_WIDTH 96
#define TILE_CORE_WIDTH 56
#define TILE_DIVISON_WIDTH 76
#define BLOCK_SIZE 32
#define MASK_SIZE 20
#define SCAN_BLOCK_SIZE 1024
#define TILE_SIZE 32

#define STAGENUM 22
#define FEATURENUM 2135


extern int stageNum;
extern int featureNum;
extern stageMeta_t *stagesMeta;
extern stage_t **stages;
extern stage_t *stagesFlat;
extern feature_t *features;


int deviceQuery();

int haarCascade(unsigned char*  outputImage, float const * image, int width, int height);
int haarAtScale(int winX, int winY, float scale, const float* integralImage, int imgWidth, int imgHeight, int winWidth, int winHeight);
float* integralImageCalc(float* image, int width, int height);

int CudaGrayScale(unsigned char* inputImage, float* grayImage, int width, int height);




#endif KERNEL_H