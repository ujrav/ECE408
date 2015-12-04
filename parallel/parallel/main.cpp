#include <stdio.h>
#include <iostream>
#include <math.h>
#include "RapidXML\rapidxml.hpp"
#include "RapidXML\rapidxml_utils.hpp"
#include "xmlparse.h"
#include "types.h"

using namespace std;
using namespace rapidxml;

unsigned char* readBMP(char* filename, int &width, int &height);
void writeBMP(char* filename, unsigned char *data, int width, int height);
int haarCascade(unsigned char*  outputImage, float const * image, int width, int height);
int haarAtScale(int winX, int winY, float scale, const float* integralImg, int imgWidth, int imgHeight, int winWidth, int winHeight);
float* integralImageCalc(float* integralImage, int width, int height);
float rectSum(float const* image, int imWidth, int inHeight, int x, int y, int w, int h);

void integralImageVerify(float* integralImage, float* imageGray, int w, int h);
void rectSumVerRect(float const* integralImage, float* image, int imWidth, int imHeight, int x, int y, int w, int h);

static int stageNum;
static stageMeta_t *stagesMeta;
static stage_t **stages;
static feature_t *features;

int main(){
	int i, j;
	int width, height;
	unsigned char *image;
	float *gray;
	unsigned char *imageGray;
	float *integralImg;
	int result = 0;

	parseClassifier("haarcascade_frontalface_alt.xml", stageNum, stagesMeta, stages, features);

	image = readBMP("besties.bmp", width, height);
	

	gray = new float[width * height];
	imageGray = new unsigned char[3 * width * height];
	for (int i = 0; i < width*height; ++i){
		gray[i] = (0.2989*((float)image[3 * i]) + 0.5870*((float)image[3 * i + 1]) + 0.1140*((float)image[3 * i + 2])) / (255.0); // in windows stored as BGR
		imageGray[3 * i] = gray[i] * 255;
		imageGray[3 * i + 1] = gray[i]* 255;
		imageGray[3 * i + 2] = gray[i]* 255;
		
		if ( i < 10)
			printf("%d %d %d \n", image[3 * i], image[3 * i + 1], image[3 * i + 2]);
		result += gray[i];
	}

	writeBMP("output.bmp", image, width, height);

	integralImg = integralImageCalc(gray, width, height);

	//integralImageVerify(integralImg, gray, width, height);

	FILE *fp;

	fp = fopen("gray.txt", "w");
	for (i = 0; i < width; i++){
		for (j = 0; j < height; j++){
			fprintf(fp, "%f ", gray[i + j*width]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	fp = fopen("grayInt.txt", "w");
	for (i = 0; i < width; i++){
		for (j = 0; j < height; j++){
			fprintf(fp, "%f ", integralImg[i + j*width]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	if (haarCascade(image, integralImg, width, height) == -1)
		cout << "Cascade failed." << endl;

	writeBMP("output.bmp", image, width, height);


	return 0;
}

unsigned char* readBMP(char* filename, int &width, int &height)
{
	int i, j;
	FILE* f = fopen(filename, "rb");
	unsigned char info[54];
	fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

	unsigned char red, green, blue;

	// extract image height and width from header
	width = *(int*)&info[18];
	height = *(int*)&info[22];

	int size = 3 * width * height;
	unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
	unsigned char* flip = new unsigned char[size];
	fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
	fclose(f);

	for (j = 0; j < height/2; j++){
		for (i = 0; i < width; i++){
			red = data[3*(i + j*width) + 2];
			green = data[3*(i + j*width) + 1];
			blue = data[3*(i + j*width)];

			data[3*(i + j*width)] = data[3*(i + (height - j - 1)*width) + 2];
			data[3*(i + j*width) + 1] = data[3*(i + (height - j - 1)*width) + 1];
			data[3*(i + j*width) + 2] = data[3*(i + (height - j - 1)*width)];

			data[3*(i + (height - j - 1)*width)] = red;
			data[3*(i + (height - j - 1)*width) + 1] = green;
			data[3*(i + (height - j - 1)*width) + 2] = blue;

		}
	}

	return data;
}

void writeBMP(char* filename, unsigned char *data, int width, int height){
	FILE *fp;
	int i, j;
	unsigned char red, green, blue;
	unsigned char header[54] = { 0 };
	unsigned char* output = new unsigned char[3 * width*height];
	fp = fopen(filename, "wb");
	header[0] = 'B';
	header[1] = 'M';
	*(int*)&header[2] = 54 + width * height * 3;
	*(int*)&header[0xA] = 54;
	*(int*)&header[0xE] = 40;

	*(int*)&header[18] = width;
	*(int*)&header[22] = height;

	header[0x1A] = 1;
	header[0x1C] = 24;
	header[0x22] = 3 * width * height;
	fwrite(header, 1, 54, fp);
	for (i = 0; i < width; i++){
		for (j = 0; j < height/2; j++){
			red = data[3 * (i + j*width)];
			green = data[3 * (i + j*width) + 1];
			blue = data[3 * (i + j*width) + 2];

			output[3 * (i + j*width) + 2] = data[3 * (i + (height - j - 1)*width)];
			output[3 * (i + j*width) + 1] = data[3 * (i + (height - j - 1)*width) + 1];
			output[3 * (i + j*width)] = data[3 * (i + (height - j - 1)*width) + 2];


			output[3 * (i + (height - j - 1)*width)] = blue;
			output[3 * (i + (height - j - 1)*width) + 1] = green;
			output[3 * (i + (height - j - 1)*width) + 2] = red;
		}
	}
	fwrite(output, 1, 3 * width*height, fp);
	fclose(fp);
}

float* integralImageCalc(float* img, int width, int height){
	float *data;

	data = new float[height*width];

	for (int i = 0; i < width; i++){
		data[i] = img[i];
	}

	for (int i = 0; i < width; i++){
		for (uint32_t j = 0; j < height; j++){
			if (j != 0){
				data[i + j * width] = data[i + (j - 1) * width] + img[i + j * width];
			}
		}
	}

	for (int i = 0; i < width; i++){
		for (int j = 0; j < height; j++){
			if (i != 0){
				data[i + j * width] = data[(i - 1) + j * width] + data[i + j * width];
			}
		}
	}

	return data;
}

void integralImageVerify(float* integralImage, float* imageGray, int w, int h){
	int i, j, x, y;
	float sum;

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

int haarCascade(unsigned char*  outputImage, const float* integralImg, int width, int height){
	int i, j;
	float scaleWidth = ((float)width) / 20.0;
	float scaleHeight = ((float)height) / 20.0;
	int step;

	float scaleStart = scaleHeight < scaleWidth ? scaleHeight : scaleWidth;

	int scaleMaxItt = ceil(log(1 / scaleStart) / log(1.0 / 1.2));

	for (int sIdx = 0; sIdx < scaleMaxItt; ++sIdx)
	{
		float scale = scaleStart*(float)powf(1.0 / 1.2, (float)(sIdx));
		cout << "Scale: " << scale << endl;

		step = scale > 2 ? scale : 2;

		int winWidth = 20 * scale;
		int winHeight = 20 * scale;

		for (int y = 0; y <= height - 1 - winHeight; y += step){
			for (int x = 0; x <= width - 1 - winWidth; x += step)
			{

				if (haarAtScale(x, y, scale, integralImg, width, height, winWidth, winHeight)){
					printf("Haar succeeded at %d, %d, %d, %d\n", x, y, winWidth, winHeight);
					for (i = 0; i < winWidth; i++){
						outputImage[3 * (x + i + (y)*width) + 1] = 255;
						outputImage[3 * (x + i + (y + winHeight - 1)*width) + 1] = 255;
					}
					for (j = 0; j < winHeight; j++){
						outputImage[3 * (x + (y + j)*width) + 1] = 255;
						outputImage[3 * (x + winWidth - 1 + (y + j)*width) + 1] = 255;
					}
				}
			}
		}
	}
	return 0;
}

int haarAtScale(int winX, int winY, float scale, const float* integralImg, int imgWidth, int imgHeight, int winWidth, int winHeight){
	// for each stage in stagesMeta
	feature_t *feature;
	float third = 0;
	for (int sIdx = 0; sIdx < stageNum; ++sIdx)
	{
		
		uint8_t stageSize = stagesMeta[sIdx].size;
		float stageThreshold = stagesMeta[sIdx].threshold;
		float featureSum = 0.0;
		float sum;

		//cout << "stage: " << sIdx << " stageSize: " << (int)stageSize << " stage thresh: " << stageThreshold << endl;

		// for each classifier in a stage
		for (int cIdx = 0; cIdx < stageSize; ++cIdx)
		{
			//if (sIdx == 3 && cIdx == 13){
				//printf("we heeeeere\n");
			//}
			// get feature index and threshold
			int fIdx = 0;
			float featureThreshold = stages[sIdx][cIdx].threshold;
			feature = &(stages[sIdx][cIdx].feature);

			// get black rectangle of feature fIdx
			uint8_t rectX = feature->black.x * scale;
			uint8_t rectY = feature->black.y * scale;
			uint8_t rectWidth = feature->black.w * scale;
			uint8_t rectHeight = feature->black.h * scale;
			int8_t rectWeight = feature->black.weight;

			float black = (float)rectWeight * rectSum(integralImg, imgWidth, imgHeight, winX + rectX, winY + rectY, rectWidth, rectHeight);
			//printf("black x:%d y:%d w:%d h:%d\n", rectX, rectY, rectWidth, rectHeight);

			// get white rectangle of feature fIdx
			rectX = feature->white.x * scale;
			rectY = feature->white.y * scale;
			rectWidth = feature->white.w * scale;
			rectHeight = feature->white.h * scale;
			rectWeight = feature->white.weight;

			float white = (float)rectWeight * rectSum(integralImg, imgWidth, imgHeight, winX + rectX, winY + rectY, rectWidth, rectHeight);
			//printf("white x:%d y:%d w:%d h:%d\n", rectX, rectY, rectWidth, rectHeight);

			third = 0;
			if (feature->third.weight){
				rectX = feature->third.x * scale;
				rectY = feature->third.y * scale;
				rectWidth = feature->third.w * scale;
				rectHeight = feature->third.h * scale;
				rectWeight = feature->third.weight;
				third = (float)rectWeight * rectSum(integralImg, imgWidth, imgHeight, winX + rectX, winY + rectY, rectWidth, rectHeight);
			}

			sum = (black + white + third) / ((float)(winWidth * winHeight));
			//printf("Feature Sum: %f, Feature Threshold: %f Black: %f White: %f\n\n", sum, featureThreshold, black, white);

			if (sum > featureThreshold)
				featureSum += stages[sIdx][cIdx].rightWeight;
			else
				featureSum += stages[sIdx][cIdx].leftWeight;

		}

		if (featureSum < stageThreshold){
			//printf("Made it to %d\n", sIdx);
			return 0;
		}

	}
	return 1;
}

float rectSum(const float* integralImage, int imWidth, int imHeight, int x, int y, int w, int h){
	float a, b, c, d;

	if (x - 1 < 0 || y - 1 < 0){
		a = 0;
	}
	else{
		a = integralImage[(x-1) + (y-1) * imWidth];
	}

	if (x - 1 + w < 0 || y - 1 < 0){
		b = 0;
	}
	else{
		b = integralImage[(x - 1 + w) + (y - 1)*imWidth];
	}

	if (x - 1 < 0 || y - 1 + h < 0){
		c = 0;
	}
	else{
		c = integralImage[(x - 1) + (y - 1 + h) * imWidth];
	}

	if (x - 1 + w < 0 || y - 1 + h < 0){
		d = 0;
	}
	else{
		d = integralImage[(x - 1 + w) + (y - 1 + h) * imWidth];
	}

	return (float)(d - c - b + a);
}

void rectSumVerRect(const float* integralImage, float* image, int imWidth, int imHeight, int x, int y, int w, int h){
	float trueSum = 0;
	int i, j, p, q;

	for (i = 0; i < w; i++){
		for (j = 0; j < w; j++){
			trueSum += image[(x + i) + (y + j)*imWidth];
		}
	}

	printf("trueSum: %f\n", trueSum);
}