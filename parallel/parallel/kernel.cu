#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "RapidXML\rapidxml.hpp"
#include "RapidXML\rapidxml_utils.hpp"
#include "bmp.h"
#include "xmlparse.h"
#include "types.h"
#include "wb.h"


using namespace std;
using namespace rapidxml;

#define TILE_WIDTH 77
#define TILE_HALO 19
#define BLOCK_SIZE 32
#define MASK_SIZE 20
#define THREAD_UTIL 9

#define STAGENUM 22
#define FEATURENUM 2135


int haarCascade(unsigned char*  outputImage, float const * image, int width, int height);
int haarAtScale(int winX, int winY, float scale, const float* integralImage, int imgWidth, int imgHeight, int winWidth, int winHeight);
float* integralImageCalc(float* integralImage, int width, int height);
__device__ __host__ float rectSum(float const* image, int imWidth, int inHeight, int x, int y, int w, int h);

int CudaHaarCascade(unsigned char*  outputImage, const float* integralImg, int width, int height);
int CudaGrayScale(unsigned char* inputImage, float* grayImage, int width, int height);

int deviceQuery();

static int stageNum;
static int featureNum;
static stageMeta_t *stagesMeta;
static stage_t **stages;
static stage_t *stagesFlat;
static feature_t *features;

__device__ __constant__ stage_t deviceStages[FEATURENUM];
__device__ __constant__ stageMeta_t deviceStagesMeta[STAGENUM];

__global__ void CudaHaarAtScale(float* deviceIntegralImage, int width, int height, int winWidth, int winHeight, float scale, float step)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x;
    int y;
	int winX, winY;
	int sIdx, cIdx;
	int i, j;
    float originX;
    float originY;
    int stageIndex;
    float a, b, c, d;
    uint8_t rectX;
    uint8_t rectY;
    uint8_t rectWidth;
    uint8_t rectHeight;
    int8_t rectWeight;
    float black;
    float white;
    float third;
    bool success;

	float featureThreshold;
	feature_t *feature;
	float sum;
	float featureSum;
	float stageThreshold;


    originX = ((float)bx)*((float)TILE_WIDTH)*scale;
	originY = ((float)by)*((float)TILE_WIDTH)*scale;

	for (i = 0; i < 3; i++){
		for (j = 0; j < 3; j++){
			winX = originX + step*(tx + i*BLOCK_SIZE);
			winY = originY + step*(ty + j*BLOCK_SIZE);

			// if (winWidth == 260 && tx == 0){
			// 	printf("%d: %d\n", bx, winX);
			// }

			if (winX <= width - winWidth && winY <= height - winWidth && 
				winX < ((float)bx + 1.0)*((float)TILE_WIDTH)*scale - winWidth && 
				winY < ((float)by + 1.0)*((float)TILE_WIDTH)*scale - winHeight){
				success = true;
				for (sIdx = 0; sIdx < STAGENUM; sIdx++){
					stageThreshold = deviceStagesMeta[sIdx].threshold;
					featureSum = 0;
					for (cIdx = 0; cIdx < deviceStagesMeta[sIdx].size; cIdx++){
						stageIndex = deviceStagesMeta[sIdx].start + cIdx;

						featureThreshold = deviceStages[stageIndex].threshold;
						feature = &(deviceStages[stageIndex].feature);

						// get black rectangle of feature fIdx
						rectX = (uint8_t)(feature->black.x * scale);
						rectY = (uint8_t)(feature->black.y * scale);
						rectWidth = (uint8_t)(feature->black.w * scale);
						rectHeight = (uint8_t)(feature->black.h * scale);
						rectWeight = feature->black.weight;

						black = rectWeight * rectSum(deviceIntegralImage, width, height, winX + rectX, winY + rectY, rectWidth, rectHeight);

						//printf("x %u y %u w %u h %u weight %d\n", rectX, rectY, rectWidth, rectHeight, rectWeight);

						// get white rectangle of feature fIdx
						rectX = (uint8_t)(feature->white.x * scale);
						rectY = (uint8_t)(feature->white.y * scale);
						rectWidth = (uint8_t)(feature->white.w * scale);
						rectHeight = (uint8_t)(feature->white.h * scale);
						rectWeight = (uint8_t)(feature->white.weight);

						white = (float)rectWeight * rectSum(deviceIntegralImage, width, height, winX + rectX, winY + rectY, rectWidth, rectHeight);

						third = 0;
						if (feature->third.weight){
							rectX = feature->third.x * scale;
							rectY = feature->third.y * scale;
							rectWidth = feature->third.w * scale;
							rectHeight = feature->third.h * scale;
							rectWeight = feature->third.weight;
							third = (float)rectWeight * rectSum(deviceIntegralImage, width, height, winX + rectX, winY + rectY, rectWidth, rectHeight);
						}

						sum = (black + white + third) / ((float)(winWidth * winHeight));

						//printf("sum: %f threshold: %f featureSum: %f right: %f left: %f\n", sum, featureThreshold, featureSum, deviceStages[stageIndex].rightWeight, deviceStages[stageIndex].leftWeight);
						//printf("black: %f white: %f third: %f\n", black, white, third);
						
						if (sum > featureThreshold)
							featureSum += deviceStages[stageIndex].rightWeight;
						else
							featureSum += deviceStages[stageIndex].leftWeight;

					}

					if (featureSum < stageThreshold){
						success = false;
						break;
					}
				}

				//if (success){
					//printf("yay %d %d %d %d\n", winX, winY, winWidth, winHeight);
				//}
			}
		}
	}
	    
}

//-------------------------------------------------------------------------
// Simple Haar Cascade Kernel (no shared memory)
//-------------------------------------------------------------------------
__global__ void simpleCudaHaar(float* deviceIntegralImage, int width, int height, int winWidth, int winHeight, float scale, int step)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int originX;
	int originY;
	int stageIndex;

	originX = tx*step;
	originY = ty*step;

	// for each stage in stagesMeta
	feature_t *feature;
	float third = 0;
	if (originX <= (width - 1 - winWidth) && originY <= (height - 1 - winHeight)) {
		for (int sIdx = 0; sIdx < STAGENUM; ++sIdx)
		{

			uint8_t stageSize = deviceStagesMeta[sIdx].size;
			float stageThreshold = deviceStagesMeta[sIdx].threshold;
			float featureSum = 0.0;
			float sum;

			// for each classifier in a stage
			for (int cIdx = 0; cIdx < stageSize; ++cIdx)
			{
				// get feature index and threshold
				stageIndex = deviceStagesMeta[sIdx].start + cIdx;
				float featureThreshold = deviceStages[stageIndex].threshold;
				feature = &(deviceStages[stageIndex].feature);

				// get black rectangle of feature fIdx
				uint8_t rectX = feature->black.x * scale;
				uint8_t rectY = feature->black.y * scale;
				uint8_t rectWidth = feature->black.w * scale;
				uint8_t rectHeight = feature->black.h * scale;
				int8_t rectWeight = feature->black.weight;

				float black = (float)rectWeight * rectSum(deviceIntegralImage, width, height, originX + rectX, originY + rectY, rectWidth, rectHeight);

				// get white rectangle of feature fIdx
				rectX = feature->white.x * scale;
				rectY = feature->white.y * scale;
				rectWidth = feature->white.w * scale;
				rectHeight = feature->white.h * scale;
				rectWeight = feature->white.weight;

				float white = (float)rectWeight * rectSum(deviceIntegralImage, width, height, originX + rectX, originY + rectY, rectWidth, rectHeight);

				third = 0;
				if (feature->third.weight){
					rectX = feature->third.x * scale;
					rectY = feature->third.y * scale;
					rectWidth = feature->third.w * scale;
					rectHeight = feature->third.h * scale;
					rectWeight = feature->third.weight;
					third = (float)rectWeight * rectSum(deviceIntegralImage, width, height, originX + rectX, originY + rectY, rectWidth, rectHeight);
				}

				sum = (black + white + third) / ((float)(winWidth * winHeight));

				if (sum > featureThreshold)
					featureSum += deviceStages[stageIndex].rightWeight;
				else
					featureSum += deviceStages[stageIndex].leftWeight;

			}

			if (featureSum < stageThreshold){
				//Failed
				return;
			}

		}
		printf("Passed at originX: %d originY: %d Width: %d Height: %d\n", originX, originY, winWidth, winHeight);
	}
	return;

}


//-------------------------------------------------------------------------
// Convert RGB to Gray Scale kernel
//-------------------------------------------------------------------------
__global__ void convertRGBToGrayScale(unsigned char *uCharInputImage, float *grayOutputImage, int width, int height)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx<width*height) {
		float r = (float)(uCharInputImage[3*idx]);
		float g = (float)(uCharInputImage[3*idx + 1]);
		float b = (float)(uCharInputImage[3*idx + 2]);

		grayOutputImage[idx] = (0.2989f*r + 0.5870f*g + 0.1140f*b) / 255.0f;
	}
}

int main(){
	int width, height;
	unsigned char *image;
	float *gray, *cudaGray;
	float *integralImg;

	deviceQuery();

	wbTime_start(Compute, "Parsing Haar Classifier");
	parseClassifierFlat("haarcascade_frontalface_alt.xml", stageNum, featureNum, stagesMeta, stagesFlat, features);
	wbTime_stop(Compute, "Parsing Haar Classifier");

	image = readBMP("besties.bmp", width, height);

	wbTime_start(Compute, "Converting serial gray scale conversion");
	gray = new float[width * height];
	convertGrayScale(image, gray, width, height);
	wbTime_stop(Compute, "Converting serial gray scale conversion");

	wbTime_start(Compute, "Converting CUDA gray scale conversion");
	cudaGray = new float[width * height];
	CudaGrayScale(image, cudaGray, width, height);
	wbTime_stop(Compute, "Converting CUDA gray scale conversion");

	wbTime_start(Compute, "Computing serial integral image");
	integralImg = integralImageCalc(cudaGray, width, height);
	wbTime_stop(Compute, "Computing serial integral image");

	wbTime_start(Compute, "Performing CUDA Haar Cascade");
	if (CudaHaarCascade(image, integralImg, width, height) == -1)
		cout << "Cascade failed." << endl;
	wbTime_stop(Compute, "Performing CUDA Haar Cascade");

	writeBMP("output.bmp", image, width, height);

	// Free my people
	//for (int s = 0; s < STAGENUM; ++s)
	//{
	//	delete[] stages[s];
	//}
	//delete[] stages;
	delete[] stagesFlat;
	delete[] stagesMeta;
	//delete[] features;

	delete[] cudaGray;
	delete[] gray;
	delete[] integralImg;
	delete[] image;

	return 0;
}



float* integralImageCalc(float* img, int width, int height){
	float *data;

	data = new float[height*width];

	for (int i = 0; i < width; i++){
		data[i] = img[i];
	}

	for (int i = 0; i < width; i++){
		for (int j = 0; j < height; j++){
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


int haarCascade(unsigned char*  outputImage, const float* integralImg, int width, int height){
	int i, j;
	float scaleWidth = ((float)width) / 20.0f;
	float scaleHeight = ((float)height) / 20.0f;
	int step;

	float scaleStart = scaleHeight < scaleWidth ? scaleHeight : scaleWidth;

	int scaleMaxItt = (int)ceil(log(1 / scaleStart) / log(1.0 / 1.2));

	for (int sIdx = 0; sIdx < scaleMaxItt; ++sIdx)
	{
		float scale = scaleStart*(float)powf((float)(1.0 / 1.2), (float)(sIdx));
		//cout << "Scale: " << scale << endl;

		step = (int)scale > 2 ? (int)scale : 2;

		int winWidth = (int)(20 * scale);
		int winHeight = (int)(20 * scale);

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

		uint16_t stageSize = stagesMeta[sIdx].size;
		float stageThreshold = stagesMeta[sIdx].threshold;
		float featureSum = 0.0;
		float sum;

		//cout << "stage: " << sIdx << " stageSize: " << (int)stageSize << " stage thresh: " << stageThreshold << endl;

		// for each classifier in a stage
		for (int cIdx = 0; cIdx < stageSize; ++cIdx)
		{
			// get feature index and threshold
			float featureThreshold = stages[sIdx][cIdx].threshold;
			feature = &(stages[sIdx][cIdx].feature);

			// get black rectangle of feature fIdx
			uint8_t rectX = (uint8_t)(feature->black.x * scale);
			uint8_t rectY = (uint8_t)(feature->black.y * scale);
			uint8_t rectWidth = (uint8_t)(feature->black.w * scale);
			uint8_t rectHeight = (uint8_t)(feature->black.h * scale);
			int8_t rectWeight = feature->black.weight;

			float black = (float)rectWeight * rectSum(integralImg, imgWidth, imgHeight, winX + rectX, winY + rectY, rectWidth, rectHeight);
			//printf("black x:%d y:%d w:%d h:%d\n", rectX, rectY, rectWidth, rectHeight);

			// get white rectangle of feature fIdx
			rectX = (uint8_t)(feature->white.x * scale);
			rectY = (uint8_t)(feature->white.y * scale);
			rectWidth = (uint8_t)(feature->white.w * scale);
			rectHeight = (uint8_t)(feature->white.h * scale);
			rectWeight = (uint8_t)(feature->white.weight);

			float white = (float)rectWeight * rectSum(integralImg, imgWidth, imgHeight, winX + rectX, winY + rectY, rectWidth, rectHeight);
			//printf("white x:%d y:%d w:%d h:%d\n", rectX, rectY, rectWidth, rectHeight);

			third = 0;
			if (feature->third.weight){
				rectX = (uint8_t)(feature->third.x * scale);
				rectY = (uint8_t)(feature->third.y * scale);
				rectWidth = (uint8_t)(feature->third.w * scale);
				rectHeight = (uint8_t)(feature->third.h * scale);
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
			//printf("Failed at %d\n", sIdx);
			return 0;
		}

	}
	// Success
	return 1;
}

__device__ __host__ float rectSum(const float* integralImage, int imWidth, int imHeight, int x, int y, int w, int h){
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

int CudaHaarCascade(unsigned char* outputImage, const float* integralImage, int width, int height){
	float scaleWidth = ((float)width) / 20.0f;
	float scaleHeight = ((float)height) / 20.0f;
	float step;
	float scale;
	cudaError_t cudaStatus;
	float *deviceIntegralImage;

	float scaleStart = scaleHeight < scaleWidth ? scaleHeight : scaleWidth;

	int scaleMaxItt = (int)ceil(log(1 / scaleStart) / log(1.0 / 1.2));

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
		fprintf(stderr, "cudaMemcpy of integralImage failed!");
	}

	for (int sIdx = 0; sIdx < scaleMaxItt; ++sIdx)
	{
		scale = scaleStart*(float)powf(1.0f / 1.2f, (float)(sIdx));
		//cout << "Scale: " << scale << endl;

		step = scale > 2.0 ? scale : 2;

		int winWidth = (int)(20 * scale);
		int winHeight = (int)(20 * scale);

		dim3 DimGrid(ceil(width / (TILE_WIDTH * scale)), ceil(height / (TILE_WIDTH * scale)), 1);
		dim3 DimGridSimple(1,1, 1);
		dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
		//printf("Kernel with dimensions %d x %d x %d launching for Scale %f\n", DimGrid.x, DimGrid.y, DimGrid.z, scale);

		CudaHaarAtScale<<<DimGrid, DimBlock>>>(deviceIntegralImage, width, height, winWidth, winHeight, scale, step);
		//simpleCudaHaar << <DimGridSimple, DimBlock >> >(deviceIntegralImage, width, height, winWidth, winHeight, scale, step);

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
		fprintf(stderr, "simpleCudaHaar launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching simpleCudaHaar!\n", cudaStatus);
	}

	cudaFree(deviceIntegralImage);

	return 0;
}

int CudaGrayScale(unsigned char* inputImage, float* grayImage, int width, int height)
{
	cudaError_t cudaStatus;
	unsigned char *deviceInputImage;
	float *deviceGrayImage;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	//allocate GPU memory
	cudaStatus = cudaMalloc((void**)&deviceInputImage, 3 * width * height * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc of deviceInputImage failed!");
	}

	cudaStatus = cudaMalloc((void**)&deviceGrayImage, width * height * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc of deviceGrayImage failed!");
	}

	cudaStatus = cudaMemcpy(deviceInputImage, inputImage, 3 * width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy of inputImage failed!");
	}

	dim3 DimGrid(ceil(width*height / float(BLOCK_SIZE)), 1, 1);
	dim3 DimBlock(BLOCK_SIZE, 1, 1);

	//printf("Kernel with dimensions %d x %d x %d launching\n", DimGrid.x, DimGrid.y, DimGrid.z);

	convertRGBToGrayScale << <DimGrid, DimBlock >> >(deviceInputImage, deviceGrayImage, width, height);

	cudaStatus = cudaMemcpy(grayImage, deviceGrayImage, width * height * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy of deviceGrayImage failed!");
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "convertRGBToGrayScale launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching convertRGBToGrayScale!\n", cudaStatus);
	}

	cudaFree(deviceGrayImage);
	cudaFree(deviceInputImage);

	return 0;
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
		printf(" Number of Multiprocessors %d\n", deviceProp.multiProcessorCount);
		printf(" Maximum global memory size: %u \n",	deviceProp.totalGlobalMem);
		printf(" Maximum constant memory size: %u\n", deviceProp.totalConstMem);
		printf(" Maximum shared memory size per block: %d\n", deviceProp.sharedMemPerBlock);
		printf(" Maximum block dimensions: %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
		printf(" Maximum grid dimensions: %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		printf(" Warp size: %d \n", deviceProp.warpSize);
	}

	return 0;
}