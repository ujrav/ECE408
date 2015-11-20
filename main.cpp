#include <stdio.h>
#include <iostream>
#include "RapidXML\rapidxml.hpp"
#include "RapidXML\rapidxml_utils.hpp"
#include "xmlparse.h"
#include "types.h"

using namespace std;
using namespace rapidxml;

unsigned char* readBMP(char* filename, int &width, int &height);
void writeBMP(char* filename, unsigned char *data, int width, int height);
int haar(unsigned char* image, int width, int height);
uint32_t* integralImageCalc(unsigned char* img, int width, int height);

static int featureNum;
static stageMeta_t *stagesMeta;
static stage_t **stages;
static feature_t *features;

int main(){
	int i, j;
	int width, height;
	unsigned char *image;
	unsigned char *gray;
	unsigned char *imageGray;
	uint32_t *integralImg;
	int result = 0;

	parseClassifier("haarcascade_frontalface_default.xml", featureNum, stagesMeta, stages, features);
	image = readBMP("margaret.bmp", width, height);

	gray = new unsigned char[width * height];
	imageGray = new unsigned char[3 * width * height];
	for (i = 0; i < width*height; i++){
		gray[i] = (image[3 * i] + image[3 * i + 1] + image[3 * i + 2]) / 3;
		imageGray[3 * i] = gray[i];
		imageGray[3 * i + 1] = gray[i];
		imageGray[3 * i + 2] = gray[i];

		result += gray[i];
	}

	integralImg = integralImageCalc(gray, width, height);

	writeBMP("gray.bmp", imageGray, width, height);

	cin >> i;
}

unsigned char* readBMP(char* filename, int &width, int &height)
{
	int i;
	FILE* f = fopen(filename, "rb");
	unsigned char info[54];
	fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

	// extract image height and width from header
	width = *(int*)&info[18];
	height = *(int*)&info[22];

	int size = 3 * width * height;
	unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
	fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
	fclose(f);

	for (i = 0; i < size; i += 3)
	{
		unsigned char tmp = data[i];
		data[i] = data[i + 2];
		data[i + 2] = tmp;
	}

	return data;
}

void writeBMP(char* filename, unsigned char *data, int width, int height){
	FILE *fp;
	unsigned char header[54] = { 0 };
	fp = fopen(filename, "w");
	printf("%x\n", fp);
	header[0] = 'B';
	header[1] = 'M';
	*(int*)&header[2] = 54 + width * height * 3;
	*(int*)&header[0xA] = 54;
	*(int*)&header[0xE] = 40;

	*(int*)&header[18] = width;
	*(int*)&header[22] = height;

	header[0x1A] = 1;
	header[0x1C] = 24;
	header[0x22] = width * height * 3;
	fwrite(header, 1, 54, fp);
	fwrite(data, 1, 3 * width*height, fp);
	fclose(fp);
}

uint32_t* integralImageCalc(unsigned char* img, int width, int height){
	uint32_t *data;
	int i, j;

	data = new uint32_t[height*width];

	for (i = 0; i < width; i++){
		data[i] = img[i];
	}

	for (i = 0; i < width; i++){
		for (j = 0; j < height; j++){
			if (j != 0){
				data[i + j * width] = data[i + (j - 1) * width] + (uint32_t)img[i + j * width];
			}
		}
	}

	for (i = 0; i < width; i++){
		for (j = 0; j < height; j++){
			if (i != 0){
				data[i + j * width] = data[(i - 1) + j * width] + data[i + j * width];
			}
		}
	}

	return data;
}

int haar(unsigned char* integralImg, int width, int height){

	return 0;
}