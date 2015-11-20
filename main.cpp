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

	if (haarCascade(integralImg, width, height, 0, 0) == -1)
		cout << "Cascade failed." << endl;


	return 0;
	//cin >> i;
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

int haarCascade(uint32_t const * integralImg, uint32_t width, uint32_t height, uint32_t winX, uint32_t winY){

	// for each stage in stagesMeta
	for (uint32_t sIdx = 0; sIdx < STAGENUM; ++sIdx)
	{

		uint8_t stageSize = stagesMeta[sIdx].size;
		float stageThreshold = stagesMeta[sIdx].threshold;
		float featureSum = 0.0;

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

			uint32_t a = integralImg[(winX + rectX) + (winY + rectY)*width];
			uint32_t b = integralImg[(winX + rectX + rectWidth) + (winY + rectY)*width];
			uint32_t c = integralImg[(winX + rectX) + (winY + rectY + rectHeight)*width];
			uint32_t d = integralImg[(winX + rectX + rectWidth) + (winY + rectY + rectHeight)*width];

			float black = float(rectWeight*(a + d - b - c));

			// get white rectangle of feature fIdx
			rectX = features[fIdx].white.x;
			rectY = features[fIdx].white.y;
			rectWidth = features[fIdx].white.w;
			rectHeight = features[fIdx].white.h;
			rectWeight = features[fIdx].white.weight;

			a = integralImg[(winX + rectX) + (winY + rectY)*width];
			b = integralImg[(winX + rectX + rectWidth) + (winY + rectY)*width];
			c = integralImg[(winX + rectX) + (winY + rectY + rectHeight)*width];
			d = integralImg[(winX + rectX + rectWidth) + (winY + rectY + rectHeight)*width];

			float white = float(rectWeight*(a + d - b - c));

			if (black + white > featureThreshold)
				featureSum += stages[sIdx][cIdx].rightWeight;
			else
				featureSum += stages[sIdx][cIdx].leftWeight;
		}

		if (featureSum < stageThreshold)
			return -1;

	}
	return 0;
}