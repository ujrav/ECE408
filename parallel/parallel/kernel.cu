#include "kernel.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "bmp.h"
#include "xmlparse.h"
#include "haar.cuh"
#include "wb.h"
#include "integralImg.cuh"

using namespace std;
using namespace rapidxml;

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
			    }                                                                      \
        } while (0)


int stageNum;
int featureNum;
stageMeta_t *stagesMeta;
stage_t **stages;
stage_t *stagesFlat;
feature_t *features;


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
	unsigned char *imageParallel;
	float *gray, *cudaGray;
	float *integralImg;

	deviceQuery();

	image = readBMP("Images/besties.bmp", width, height);
	imageParallel = new unsigned char[width * height * 3];
	memcpy(imageParallel, image, width*height * 3);

	cout << "Image dimensions: " << width << " x " << height << endl;

	wbTime_start(Compute, "Parsing Haar Classifier");
	parseClassifierFlat("haarcascade_frontalface_alt.xml", stageNum, featureNum, stagesMeta, stagesFlat, features);
	wbTime_stop(Compute, "Parsing Haar Classifier");

	gray = new float[width * height];
	wbTime_start(Compute, "Converting serial gray scale conversion");
	convertGrayScale(image, gray, width, height);
	wbTime_stop(Compute, "Converting serial gray scale conversion");

	cudaGray = new float[width * height];
	wbTime_start(Compute, "Converting CUDA gray scale conversion");
	CudaGrayScale(image, cudaGray, width, height);
	wbTime_stop(Compute, "Converting CUDA gray scale conversion");

	wbTime_start(Compute, "Computing serial integral image");
	integralImg = integralImageCalc(cudaGray, width, height);
	wbTime_stop(Compute, "Computing serial integral image");

	wbTime_start(Compute, "Performing CUDA Haar Cascade");
	if (CudaHaarCascade(imageParallel, integralImg, width, height) == -1)
		cout << "Cascade failed." << endl;
	wbTime_stop(Compute, "Performing CUDA Haar Cascade");

	writeBMP("Images/output.bmp", imageParallel, width, height);

	delete[] stagesMeta;

	wbTime_start(Compute, "Parsing Haar Classifier");
	parseClassifier("haarcascade_frontalface_alt.xml", stageNum, stagesMeta, stages, features);
	wbTime_stop(Compute, "Parsing Haar Classifier");

	wbTime_start(Compute, "Performing serial Haar Cascade");
	if (haarCascade(image, integralImg, width, height) == -1)
		cout << "Cascade failed." << endl;
	wbTime_stop(Compute, "Performing serial Haar Cascade");

	writeBMP("Images/outputSerial.bmp", image, width, height);

	// Free my people
	for (int s = 0; s < STAGENUM; ++s)
	{
		delete[] stages[s];
	}
	delete[] stages;
	delete[] stagesFlat;
	delete[] stagesMeta;
	//delete[] features;

	delete[] cudaGray;
	delete[] gray;
	delete[] integralImg;
	delete[] image;
	delete[] imageParallel;

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
					//printf("Haar succeeded at %d, %d, %d, %d\n", x, y, winWidth, winHeight);
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
			int rectX = (int)(feature->black.x * scale);
			int rectY = (int)(feature->black.y * scale);
			int rectWidth = (int)(feature->black.w * scale);
			int rectHeight = (int)(feature->black.h * scale);
			int8_t rectWeight = feature->black.weight;

			float black = (float)rectWeight * rectSum(integralImg, imgWidth, imgHeight, winX + rectX, winY + rectY, rectWidth, rectHeight);
			//printf("black x:%d y:%d w:%d h:%d\n", rectX, rectY, rectWidth, rectHeight);

			// get white rectangle of feature fIdx
			rectX = (int)(feature->white.x * scale);
			rectY = (int)(feature->white.y * scale);
			rectWidth = (int)(feature->white.w * scale);
			rectHeight = (int)(feature->white.h * scale);
			rectWeight = (int)(feature->white.weight);

			float white = (float)rectWeight * rectSum(integralImg, imgWidth, imgHeight, winX + rectX, winY + rectY, rectWidth, rectHeight);
			//printf("white x:%d y:%d w:%d h:%d\n", rectX, rectY, rectWidth, rectHeight);

			third = 0;
			if (feature->third.weight){
				rectX = (int)(feature->third.x * scale);
				rectY = (int)(feature->third.y * scale);
				rectWidth = (int)(feature->third.w * scale);
				rectHeight = (int)(feature->third.h * scale);
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
		cout << " Maximum global memory size: " << deviceProp.totalGlobalMem << endl;
		//printf(" Maximum global memory size: %d \n",	deviceProp.totalGlobalMem);
		printf(" Maximum constant memory size: %d\n", deviceProp.totalConstMem);
		printf(" Maximum shared memory size per block: %d\n", deviceProp.sharedMemPerBlock);
		printf(" Maximum block dimensions: %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
		printf(" Maximum grid dimensions: %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		printf(" Warp size: %d \n", deviceProp.warpSize);
	}

	return 0;
}