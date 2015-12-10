#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "RapidXML\rapidxml.hpp"
#include "RapidXML\rapidxml_utils.hpp"
#include "xmlparse.h"
#include "types.h"

using namespace std;
using namespace rapidxml;

#define TILE_WIDTH 96
#define TILE_CORE_WIDTH 56
#define TILE_DIVISON_WIDTH 76
#define BLOCK_SIZE 32
#define MASK_SIZE 20

#define STAGENUM 22
#define FEATURENUM 2135

unsigned char* readBMP(char* filename, int &width, int &height);
void writeBMP(char* filename, unsigned char *data, int width, int height);
int haarCascade(unsigned char*  outputImage, float const * image, int width, int height);
int haarAtScale(int winX, int winY, float scale, const float* integralImage, int imgWidth, int imgHeight, int winWidth, int winHeight);
float* integralImageCalc(float* integralImage, int width, int height);
float rectSum(float const* image, int imWidth, int inHeight, int x, int y, int w, int h);


void integralImageVerify(float* integralImage, float* imageGray, int w, int h);
void rectSumVerRect(float const* integralImage, float* image, int imWidth, int imHeight, int x, int y, int w, int h);

int CudaHaarCascade(unsigned char*  outputImage, const float* integralImg, int width, int height);

int deviceQuery();

static int stageNum;
static int featureNum;
static stageMeta_t *stagesMeta;
static stage_t **stages;
static stage_t *stagesFlat;
static feature_t *features;

__device__ __constant__ stage_t deviceStages[FEATURENUM];
__device__ __constant__ stageMeta_t deviceStagesMeta[STAGENUM];

__global__ void CudaHaarAtScale(float* deviceIntegralImage, int width, int height, int winWidth, int winHeight, float scale, int step)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x;
    int y;
    int originX;
    int originY;
    int stageIndex;
    float a, b, c, d;


    originX = ((float)bx)*((float)TILE_DIVISON_WIDTH)*scale;
    originY = ((float)by)*((float)TILE_DIVISON_WIDTH)*scale;

	for (int i = -1 * (MASK_SIZE*scale); i < TILE_CORE_WIDTH * scale; i += step*BLOCK_SIZE){
		for (int j = -1 * (MASK_SIZE*scale); j < TILE_CORE_WIDTH * scale; j += step*BLOCK_SIZE){
    		x = originX + (((float)tx) * scale) + i;
    		y = originY + (((float)ty) * scale) + j;
			if (x >= 0 && x <= width - winWidth - 1 && y >= 0 && y <= height - winHeight - 1 && x == 24 && y == 20 && winWidth == 81){
				printf("%f %d %d testing window at %d %d %d %d %d %d\n",scale, i, j, x, y, bx, by, tx, ty);
				feature_t *feature;
				float third = 0;
				bool result = true;
				for (int sIdx = 0; sIdx < STAGENUM; ++sIdx)
				{
					uint8_t stageSize = deviceStagesMeta[sIdx].size;
					float stageThreshold = deviceStagesMeta[sIdx].threshold;
					float featureSum = 0.0;
					float sum;

					//cout << "stage: " << sIdx << " stageSize: " << (int)stageSize << " stage thresh: " << stageThreshold << endl;

					// for each classifier in a stage
					for (int cIdx = 0; cIdx < stageSize; ++cIdx)
					{
						stageIndex = deviceStagesMeta[sIdx].start + cIdx;
						// get feature index and threshold
						int fIdx = 0;
						float featureThreshold = deviceStages[stageIndex].threshold;
						feature = &(deviceStages[stageIndex].feature);

						// get black rectangle of feature fIdx
						uint8_t rectX = feature->black.x * scale + x;
						uint8_t rectY = feature->black.y * scale + y;
						uint8_t rectWidth = feature->black.w * scale;
						uint8_t rectHeight = feature->black.h * scale;
						int8_t rectWeight = feature->black.weight;

						if (rectX - 1 < 0 || rectY - 1 < 0){
							a = 0;
						}
						else{
							a = deviceIntegralImage[(rectX - 1) + (rectY - 1) * width];
						}

						if (rectX - 1 + rectWidth< 0 || rectY - 1 < 0){
							b = 0;
						}
						else{
							b = deviceIntegralImage[(rectX - 1 + rectWidth) + (rectY - 1)*width];
						}

						if (rectX - 1 < 0 || y - 1 + rectHeight < 0){
							c = 0;
						}
						else{
							c = deviceIntegralImage[(rectX - 1) + (rectY - 1 + rectHeight) * width];
						}

						if (rectX - 1 + rectWidth< 0 || y - 1 + rectHeight < 0){
							d = 0;
						}
						else{
							d = deviceIntegralImage[(rectX - 1 + rectWidth) + (rectY - 1 + rectHeight) * width];
						}
						float black = d + a - b - c;
						//printf("black x:%d y:%d w:%d h:%d\n", rectX, rectY, rectWidth, rectHeight);

						// get white rectangle of feature fIdx
						rectX = feature->white.x * scale + x;
						rectY = feature->white.y * scale + y;
						rectWidth = feature->white.w * scale;
						rectHeight = feature->white.h * scale;
						rectWeight = feature->white.weight;

						if (rectX - 1 < 0 || rectY - 1 < 0){
							a = 0;
						}
						else{
							a = deviceIntegralImage[(rectX - 1) + (rectY - 1) * width];
						}

						if (rectX - 1 + rectWidth< 0 || rectY - 1 < 0){
							b = 0;
						}
						else{
							b = deviceIntegralImage[(rectX - 1 + rectWidth) + (rectY - 1)*width];
						}

						if (rectX - 1 < 0 || y - 1 + rectHeight < 0){
							c = 0;
						}
						else{
							c = deviceIntegralImage[(rectX - 1) + (rectY - 1 + rectHeight) * width];
						}

						if (rectX - 1 + rectWidth< 0 || y - 1 + rectHeight < 0){
							d = 0;
						}
						else{
							d = deviceIntegralImage[(rectX - 1 + rectWidth) + (rectY - 1 + rectHeight) * width];
						}
						float white = d + a - b - c;

						third = 0;
						if (feature->third.weight){
							rectX = feature->third.x * scale;
							rectY = feature->third.y * scale;
							rectWidth = feature->third.w * scale;
							rectHeight = feature->third.h * scale;
							rectWeight = feature->third.weight;
							if (rectX - 1 < 0 || rectY - 1 < 0){
								a = 0;
							}
							else{
								a = deviceIntegralImage[(rectX - 1) + (rectY - 1) * width];
							}

							if (rectX - 1 + rectWidth< 0 || rectY - 1 < 0){
								b = 0;
							}
							else{
								b = deviceIntegralImage[(rectX - 1 + rectWidth) + (rectY - 1)*width];
							}

							if (rectX - 1 < 0 || y - 1 + rectHeight < 0){
								c = 0;
							}
							else{
								c = deviceIntegralImage[(rectX - 1) + (rectY - 1 + rectHeight) * width];
							}

							if (rectX - 1 + rectWidth< 0 || y - 1 + rectHeight < 0){
								d = 0;
							}
							else{
								d = deviceIntegralImage[(rectX - 1 + rectWidth) + (rectY - 1 + rectHeight) * width];
							}
							float third = d + a - b - c;
						}

						sum = (black + white + third) / ((float)(winWidth * winHeight));
						

						if (sum > featureThreshold)
							featureSum += deviceStages[stageIndex].rightWeight;
						else
							featureSum += deviceStages[stageIndex].leftWeight;

						printf("Feature rect Sum: %f, Feature Threshold: %f Black: %f White: %f\n", sum, featureThreshold, black, white);
						printf("Feature Sum %d on stage %d: %f, Stage Threshold: %f\n\n", cIdx, sIdx, featureSum, stageThreshold);

					}

					if (featureSum < stageThreshold){
						result = false;
						break;
					}

				}

				if (result){
					printf("Holy shit success?");
				}

			}
    	}
    }

    
}

int main(){
	int i, j;
	int width, height;
	unsigned char *image;
	float *gray;
	unsigned char *imageGray;
	float *integralImg;
	int result = 0;
	

	parseClassifierFlat("haarcascade_frontalface_alt.xml", stageNum, featureNum, stagesMeta, stagesFlat, features);

	deviceQuery();

	

	image = readBMP("margaretBig.bmp", width, height);


	gray = new float[width * height];
	imageGray = new unsigned char[3 * width * height];
	for (int i = 0; i < width*height; ++i){
		gray[i] = (0.2989*((float)image[3 * i]) + 0.5870*((float)image[3 * i + 1]) + 0.1140*((float)image[3 * i + 2])) / (255.0); // in windows stored as BGR
		imageGray[3 * i] = gray[i] * 255;
		imageGray[3 * i + 1] = gray[i] * 255;
		imageGray[3 * i + 2] = gray[i] * 255;

		result += gray[i];
	}

	integralImg = integralImageCalc(gray, width, height);

	//FILE *fp;

	//fp = fopen("gray.txt", "w");
	//for (i = 0; i < width; i++){
	//	for (j = 0; j < height; j++){
	//		fprintf(fp, "%f ", gray[i + j*width]);
	//	}
	//	fprintf(fp, "\n");
	//}
	//fclose(fp);

	//fp = fopen("grayInt.txt", "w");
	//for (i = 0; i < width; i++){
	//	for (j = 0; j < height; j++){
	//		fprintf(fp, "%f ", integralImg[i + j*width]);
	//	}
	//	fprintf(fp, "\n");
	//}
	//fclose(fp);


	if (CudaHaarCascade(image, integralImg, width, height) == -1)
		cout << "Cascade failed." << endl;

	//writeBMP("output.bmp", image, width, height);

	int a;
	cin >> a;

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

	for (j = 0; j < height / 2; j++){
		for (i = 0; i < width; i++){
			red = data[3 * (i + j*width) + 2];
			green = data[3 * (i + j*width) + 1];
			blue = data[3 * (i + j*width)];

			data[3 * (i + j*width)] = data[3 * (i + (height - j - 1)*width) + 2];
			data[3 * (i + j*width) + 1] = data[3 * (i + (height - j - 1)*width) + 1];
			data[3 * (i + j*width) + 2] = data[3 * (i + (height - j - 1)*width)];

			data[3 * (i + (height - j - 1)*width)] = red;
			data[3 * (i + (height - j - 1)*width) + 1] = green;
			data[3 * (i + (height - j - 1)*width) + 2] = blue;

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
		for (j = 0; j < height / 2; j++){
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

int CudaHaarCascade(unsigned char*  outputImage, const float* integralImage, int width, int height){
	int i, j;
	float scaleWidth = ((float)width) / 20.0;
	float scaleHeight = ((float)height) / 20.0;
	int step;
	float scale;
	cudaError_t cudaStatus;
	float *deviceIntegralImage;

	float scaleStart = scaleHeight < scaleWidth ? scaleHeight : scaleWidth;

	int scaleMaxItt = ceil(log(1 / scaleStart) / log(1.0 / 1.2));

	cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    //allocate GPU memory
    cudaStatus = cudaMalloc((void**)&deviceIntegralImage, height * width * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc of integralImage failed!");
    }

    // copy data to GPU memory
	cudaStatus = cudaMemcpyToSymbol(deviceStagesMeta, stagesMeta, stageNum * sizeof(stageMeta_t), 0, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy of StageMeta failed!");
    }

	cudaStatus = cudaMemcpyToSymbol(deviceStages, stagesFlat, featureNum * sizeof(stage_t), 0, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy of StageMeta failed!");
	}

	cudaStatus = cudaMemcpy(deviceIntegralImage, integralImage, height * width * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

	for (int sIdx = 0; sIdx < scaleMaxItt; ++sIdx)
	{
		scale = scaleStart*(float)powf(1.0 / 1.2, (float)(sIdx));
		cout << "Scale: " << scale << endl;

		step = scale > 2 ? scale : 2;

		int winWidth = 20 * scale;
		int winHeight = 20 * scale;

		dim3 DimGrid(width / ((int)(((float)TILE_DIVISON_WIDTH) * scale)), height / ((int)(((float)TILE_DIVISON_WIDTH) * scale)), 1);
		if (width % ((int)(((float)TILE_DIVISON_WIDTH) * scale))) DimGrid.x++;
		if (height % ((int)(((float)TILE_DIVISON_WIDTH) * scale))) DimGrid.y++;

	    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

	    printf("Kernel with dimensions %d x %d x %d launching\n", DimGrid.x, DimGrid.y, DimGrid.z);

		CudaHaarAtScale<<<DimGrid, DimBlock>>>(deviceIntegralImage, width, height, winWidth, winHeight, scale, step);

		// for (int y = 0; y <= height - 1 - winHeight; y += step){
		// 	for (int x = 0; x <= width - 1 - winWidth; x += step)
		// 	{

		// 		if (haarAtScale(x, y, scale, integralImg, width, height, winWidth, winHeight)){
		// 			printf("Haar succeeded at %d, %d, %d, %d\n", x, y, winWidth, winHeight);
		// 			for (i = 0; i < winWidth; i++){
		// 				outputImage[3 * (x + i + (y)*width) + 1] = 255;
		// 				outputImage[3 * (x + i + (y + winHeight - 1)*width) + 1] = 255;
		// 			}
		// 			for (j = 0; j < winHeight; j++){
		// 				outputImage[3 * (x + (y + j)*width) + 1] = 255;
		// 				outputImage[3 * (x + winWidth - 1 + (y + j)*width) + 1] = 255;
		// 			}
		// 		}
		// 	}
		// }
	}
	// Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
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
		a = integralImage[(x - 1) + (y - 1) * imWidth];
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

int deviceQuery(){
	int deviceCount;

	cudaGetDeviceCount(&deviceCount);

	for (int dev = 0; dev < deviceCount; dev++) {
		cudaDeviceProp deviceProp;

		cudaGetDeviceProperties(&deviceProp, dev);

		if (dev == 0) {
			if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
				printf("No CUDA GPU has been detected\n");
				return -1;
			}
			else if (deviceCount == 1) {
				//@@ WbLog is a provided logging API (similar to Log4J).
				//@@ The logging function wbLog takes a level which is either
				//@@ OFF, FATAL, ERROR, WARN, INFO, DEBUG, or TRACE and a
				//@@ message to be printed.
				printf("There is 1 device supporting CUDA\n");
			}
			else {
				printf("There are %d devices supporting CUDA\n", deviceCount);
			}
		}

		printf("Device %s name: %s\n",dev, deviceProp.name);
		printf(" Computational Capabilities: %d.%d\n", deviceProp.major, deviceProp.minor);
		printf(" Maximum global memory size: %d \n",	deviceProp.totalGlobalMem);
		printf(" Maximum constant memory size: %d\n", deviceProp.totalConstMem);
		printf(" Maximum shared memory size per block: %d\n", deviceProp.sharedMemPerBlock);
		printf(" Maximum block dimensions: %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
		printf(" Maximum grid dimensions: %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		printf(" Warp size: %d \n", deviceProp.warpSize);
	}

	return 0;
}