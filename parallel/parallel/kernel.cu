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

#define TILE_WIDTH 96
#define TILE_CORE_WIDTH 56
#define TILE_DIVISON_WIDTH 76
#define BLOCK_SIZE 32
#define MASK_SIZE 20
#define SCAN_BLOCK_SIZE 1024

#define STAGENUM 22
#define FEATURENUM 2135

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
	    }                                                                      \
    } while (0)


int haarCascade(unsigned char*  outputImage, float const * image, int width, int height);
int haarAtScale(int winX, int winY, float scale, const float* integralImage, int imgWidth, int imgHeight, int winWidth, int winHeight);
float* integralImageCalc(float* image, int width, int height);
__device__ __host__ float rectSum(float const* image, int imWidth, int inHeight, int x, int y, int w, int h);

int CudaHaarCascade(unsigned char*  outputImage, const float* integralImg, int width, int height);
int CudaGrayScale(unsigned char* inputImage, float* grayImage, int width, int height);
int CudaIntegralImage(float* grayImage, float* integralImage, int width, int height);

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
							third = d + a - b - c;
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
		printf("Passed at originX: %d originY: %d\n", originX, originY);
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

		grayOutputImage[idx] = (0.00117f*r + 0.0023f*g + 0.00045f*b); //(0.2989f*r + 0.5870f*g + 0.1140f*b) / 255.0f;
	}
}


//-------------------------------------------------------------------------
// Parallel Scan Add Auxiliary Kernel (Part 3)
//-------------------------------------------------------------------------
__global__ void parallelScanAddAux(float *aux, float *output, int len) {

	int tx = threadIdx.x;
	int bdx = blockDim.x;
	int bix = blockIdx.x;
	int start = 2 * bdx*bix;
	int idx = start + tx;

	// add aux[x] to scan block x+1, ignoring block 0 since it's done
	if (bix != 0)
	{
		if (idx<len)
		{
			output[idx] += aux[bix - 1];
		}
		if (idx + bdx<len)
		{
			output[idx + bdx] += aux[bix - 1];
		}
	}


}

//-------------------------------------------------------------------------
// Parallel Scan Kernel (Part 1 & 2)
//-------------------------------------------------------------------------
__global__ void scanRow(float *input, float *output, float *aux, int len) {
	// LECTURE METHOD with 2 reads and 2 writes
	__shared__ float scanBlock[BLOCK_SIZE << 1];

	int tx = threadIdx.x;
	int bdx = blockDim.x;
	int biy = blockIdx.y;
	int bix = blockIdx.x;
	int start = 2 * bdx*bix;
	int ix = start + tx;

	// global reads
	scanBlock[tx] = ix<len ? input[biy*len + ix] : 0.0;
	scanBlock[bdx + tx] = ix + bdx<len ? input[biy*len + ix + bdx] : 0.0;

	// reduction phase
	for (int stride = 1; stride <= bdx; stride <<= 1)
	{
		__syncthreads();
		int index = (tx + 1)*stride * 2 - 1;
		if (index<2 * bdx)
		{
			scanBlock[index] += scanBlock[index - stride];
		}
	}

	// post reduction reverse phase
	for (int stride = bdx >> 1; stride>0; stride >>= 1)
	{
		__syncthreads();
		int index = (tx + 1)*stride * 2 - 1;
		if (index + stride<2 * bdx)
		{
			scanBlock[index + stride] += scanBlock[index];
		}
	}

	// global writes
	__syncthreads();
	if (ix<len)
	{
		output[biy*len + ix] = scanBlock[tx];
	}

	if (ix + bdx<len)
	{
		output[biy*len + ix + bdx] = scanBlock[tx + bdx];
	}

	// each thread block writes its entire sum (last element)
	// into the auxiliary output according to its block ID
	if (aux != NULL && tx == 0)
	{
		aux[bix] = scanBlock[2 * BLOCK_SIZE - 1];
	}

}

int main(){
	int width, height;
	unsigned char *image;
	float *gray, *cudaGray;
	float *integralImg, *cudaIntegralImg;

	deviceQuery();

	image = readBMP("besties.bmp", width, height);

	cout << "Image dimensions: " << width << " x " << height << endl;

	wbTime_start(Compute, "Parsing Haar Classifier");
	parseClassifierFlat("haarcascade_frontalface_alt.xml", stageNum, featureNum, stagesMeta, stagesFlat, features);
	wbTime_stop(Compute, "Parsing Haar Classifier");

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

	wbTime_start(Compute, "Computing CUDA integral image");
	cudaIntegralImg = new float[width * height];
	CudaIntegralImage(cudaGray, cudaIntegralImg, width, height);
	wbTime_stop(Compute, "Computing CUDA integral image");

	for (int i = 0; i < 10; ++i)
	{
		cout << "cudaGray: " << cudaGray[i] << " cudaIntegralImg: " << cudaIntegralImg[i] << " integralImg : " << integralImg[i] << endl;
	}

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

	wbTime_start(Compute, "Performing CUDA Haar Cascade");
	if (CudaHaarCascade(image, integralImg, width, height) == -1)
		cout << "Cascade failed." << endl;
	wbTime_stop(Compute, "Performing CUDA Haar Cascade");

	delete stagesMeta;

	wbTime_start(Compute, "Parsing Haar Classifier");
	parseClassifier("haarcascade_frontalface_alt.xml", stageNum, stagesMeta, stages, features);
	wbTime_stop(Compute, "Parsing Haar Classifier");

	wbTime_start(Compute, "Performing serial Haar Cascade");
	if (haarCascade(image, integralImg, width, height) == -1)
		cout << "Cascade failed." << endl;
	wbTime_stop(Compute, "Performing serial Haar Cascade");

	writeBMP("output.bmp", image, width, height);

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
	int step;
	float scale;
	cudaError_t cudaStatus;
	float *deviceIntegralImage;

	float scaleStart = scaleHeight < scaleWidth ? scaleHeight : scaleWidth;

	int scaleMaxItt = (int)ceil(log(1 / scaleStart) / log(1.0 / 1.2));

	cout << scaleMaxItt << endl;

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
		cout << "Scale: " << scale << endl;

		step = (int)scale > 2 ? (int)scale : 2;

		int winWidth = (int)(20 * scale);
		int winHeight = (int)(20 * scale);

		//dim3 DimGrid(width / ((int)(((float)TILE_DIVISON_WIDTH) * scale)), height / ((int)(((float)TILE_DIVISON_WIDTH) * scale)), 1);
		//if (width % ((int)(((float)TILE_DIVISON_WIDTH) * scale))) DimGrid.x++;
		//if (height % ((int)(((float)TILE_DIVISON_WIDTH) * scale))) DimGrid.y++;

		dim3 DimGridSimple(1, 1, 1);
		dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

		//printf("Kernel with dimensions %d x %d x %d launching\n", DimGridSimple.x, DimGridSimple.y, DimGridSimple.z);

		//CudaHaarAtScale<<<DimGrid, DimBlock>>>(deviceIntegralImage, width, height, winWidth, winHeight, scale, step);
		simpleCudaHaar << <DimGridSimple, DimBlock >> >(deviceIntegralImage, width, height, winWidth, winHeight, scale, step);

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

int CudaIntegralImage(float* grayImage, float* integralImage, int width, int height)
{
	cudaError_t cudaStatus;
	float *deviceInputImage;
	float *deviceIntegralImage;
	float *deviceAuxInput;
	float *deviceAuxOutput;
	int numElements; // number of elements in the list
	int gridCols, gridRows;	// grid dimensions

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	numElements = width;
	gridCols = ceil(numElements / float(SCAN_BLOCK_SIZE << 1));
	gridRows = height;

	//allocate GPU memory
	wbCheck(cudaMalloc((void **)&deviceInputImage, width * height * sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceIntegralImage, width * height * sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceAuxInput, gridCols * sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceAuxOutput, gridCols * sizeof(float)));

	//cudaStatus = cudaMalloc((void**)&deviceInputImage, width * height * sizeof(float));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc of deviceInputImage failed!");
	//}

	//cudaStatus = cudaMalloc((void**)&deviceIntegralImage, width * height * sizeof(float));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc of deviceGrayImage failed!");
	//}

	cudaStatus = cudaMemcpy(deviceInputImage, grayImage, width * height * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy of inputImage failed!");
	}

	dim3 DimGrid(gridCols, gridRows, 1);
	dim3 DimBlock(SCAN_BLOCK_SIZE, 1, 1);
	dim3 DimGrid2(1, 1, 1);

	//for (int r = 0; r < height; ++r)
	{
		wbCheck(cudaMemset(deviceAuxInput, 0, gridCols * sizeof(float)));
		wbCheck(cudaMemset(deviceAuxOutput, 0, gridCols * sizeof(float)));
		//printf("Kernel with dimensions %d x %d x %d launching\n", DimGrid.x, DimGrid.y, DimGrid.z);

		scanRow << < DimGrid, DimBlock >> > (deviceInputImage, deviceIntegralImage, deviceAuxInput, numElements);	// Part 1
		wbCheck(cudaDeviceSynchronize());
		scanRow << < DimGrid2, DimBlock >> > (deviceAuxInput, deviceAuxOutput, NULL, gridCols);					// Part 2
		wbCheck(cudaDeviceSynchronize());
		parallelScanAddAux << < DimGrid, DimBlock >> > (deviceAuxOutput, deviceIntegralImage, numElements);				// Part 3
		wbCheck(cudaDeviceSynchronize());
	}


	cudaStatus = cudaMemcpy(integralImage, deviceIntegralImage, width * height * sizeof(float), cudaMemcpyDeviceToHost);
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

	cudaFree(deviceIntegralImage);
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