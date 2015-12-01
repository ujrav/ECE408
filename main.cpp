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
int haarCascade(uint32_t const * image, uint32_t width, uint32_t height, uint32_t winX, uint32_t winY);
uint32_t* integralImageCalc(unsigned char* integralImage, uint32_t width, uint32_t height);
float rectSum(uint32_t const* image, int imWidth, int inHeight, int x, int y, int w, int h);

void integralImageVerify(uint32_t* integralImage, unsigned char* imageGray, int w, int h);

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
	for (int i = 0; i < width*height; ++i){
		gray[i] = (image[3 * i] + image[3 * i + 1] + image[3 * i + 2]) / 3;
		imageGray[3 * i] = gray[i];
		imageGray[3 * i + 1] = gray[i];
		imageGray[3 * i + 2] = gray[i];

		result += gray[i];
	}

	integralImg = integralImageCalc(gray, width, height);

	writeBMP("gray.bmp", imageGray, width, height);

	integralImageVerify(integralImg, gray, width, height);

	FILE *fp;

	fp = fopen("gray.txt", "w");
	for (i = 0; i < width; i++){
		for (j = 0; j < height; j++){
			fprintf(fp, "%d ", gray[i + j*width]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	fp = fopen("grayInt.txt", "w");
	for (i = 0; i < width; i++){
		for (j = 0; j < height; j++){
			fprintf(fp, "%5d ", integralImg[i + j*width]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	if (haarCascade(integralImg, width, height, 0, 0) == -1)
		cout << "Cascade failed." << endl;

	cin >> i;

	return 0;
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

uint32_t* integralImageCalc(unsigned char* img, uint32_t width, uint32_t height){
	uint32_t *data;

	data = new uint32_t[height*width];

	for (uint32_t i = 0; i < width; i++){
		data[i] = (uint32_t)img[i];
	}

	for (uint32_t i = 0; i < width; i++){
		for (uint32_t j = 0; j < height; j++){
			if (j != 0){
				data[i + j * width] = data[i + (j - 1) * width] + (uint32_t)img[i + j * width];
			}
		}
	}

	for (uint32_t i = 0; i < width; i++){
		for (uint32_t j = 0; j < height; j++){
			if (i != 0){
				data[i + j * width] = data[(i - 1) + j * width] + data[i + j * width];
			}
		}
	}

	return data;
}

void integralImageVerify(uint32_t* integralImage, unsigned char* imageGray, int w, int h){
	int i, j, x, y;
	int sum;

	for (x = 0; x < w; x++){
		for (y = 0; y < h; y++){
			sum = 0;
			for (i = 0; i <= x; i++){
				for (j = 0; j <= y; j++){
					sum += imageGray[i + j*w];
				}
			}
			if (sum != integralImage[x + y * w]){
				printf("integral image failed at %d %d\n sum = %d integralimage = %d\n", x, y, sum, integralImage[x + y * w]);
			}
		}
	}
}

int haarCascade(uint32_t const * integralImg, uint32_t width, uint32_t height, uint32_t winX, uint32_t winY){

	// for each stage in stagesMeta
	for (uint32_t sIdx = 0; sIdx < STAGENUM; ++sIdx)
	{

		uint8_t stageSize = stagesMeta[sIdx].size;
		float stageThreshold = stagesMeta[sIdx].threshold;
		float featureSum = 0.0;
		float sum;

		cout << "stage: " << sIdx << " stageSize: " << (uint32_t)stageSize << " stage thresh: " << stageThreshold << endl;

		// for each classifier in a stage
		for (uint32_t cIdx = 0; cIdx < stageSize; ++cIdx)
		{
			// get feature index and threshold
			int fIdx = stages[sIdx][cIdx].featureIdx;
			float featureThreshold = stages[sIdx][cIdx].threshold;

			// get black rectangle of feature fIdx
			uint8_t rectX = features[fIdx].black.x;
			uint8_t rectY = features[fIdx].black.y;
			uint8_t rectWidth = features[fIdx].black.w;
			uint8_t rectHeight = features[fIdx].black.h;
			int8_t rectWeight = features[fIdx].black.weight;

			float black = (float)rectWeight * rectSum(integralImg, width, height, winX + rectX, winY + rectY, rectWidth, rectHeight);
			//printf("black a: %d b: %d c: %d d: %d  x:%d y:%d w:%d h:%d\n", a, b, c, d, rectX, rectY, rectWidth, rectHeight);

			// get white rectangle of feature fIdx
			rectX = features[fIdx].white.x;
			rectY = features[fIdx].white.y;
			rectWidth = features[fIdx].white.w;
			rectHeight = features[fIdx].white.h;
			rectWeight = features[fIdx].white.weight;

			float white = (float)rectWeight * rectSum(integralImg, width, height, winX + rectX, winY + rectY, rectWidth, rectHeight);
			//printf("white a: %d b: %d c: %d d: %d  x:%d y:%d w:%d h:%d\n", a, b, c, d, rectX, rectY, rectWidth, rectHeight);

			sum = (black + white) / (24.0*24.0);

			printf("Feature Sum: %f, Feature Threshold: %f Black: %f White: %f\n\n", sum, featureThreshold, black, white);
			

			if (sum > featureThreshold)
				featureSum += stages[sIdx][cIdx].rightWeight;
			else
				featureSum += stages[sIdx][cIdx].leftWeight;

			printf("featureSum %f \n", featureSum);
		}

		if (featureSum < stageThreshold)
			return -1;

	}
	return 0;
}

float rectSum(uint32_t const* image, int imWidth, int inHeight, int x, int y, int w, int h){
	uint32_t a, b, c, d;

	if (x - 1 < 0 || y - 1 < 0){
		a = 0;
	}
	else{
		a = image[(x-1) + (y-1) * imWidth];
	}

	if (x - 1 + w < 0 || y - 1 < 0){
		b = 0;
	}
	else{
		b = image[(x - 1 + w) + (y - 1)*imWidth];
	}

	if (x - 1 < 0 || y - 1 + h < 0){
		c = 0;
	}
	else{
		c = image[(x - 1) + (y - 1 + h) * imWidth];
	}

	if (x - 1 + w < 0 || y - 1 + h < 0){
		d = 0;
	}
	else{
		d = image[(x - 1 + w) + (y - 1 + h) * imWidth];
	}

	return (float)(d - c - b + a);
}
