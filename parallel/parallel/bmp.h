#ifndef BMP_H
#define BMP_H

#include "types.h"

unsigned char* readBMP(char* filename, int &width, int &height);
void writeBMP(char* filename, unsigned char *data, int width, int height);
void convertGrayScale(unsigned char *inputImg, float *outputImg, int imgWidth, int imgHeight);


#endif BMP_H