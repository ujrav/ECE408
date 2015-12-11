// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... +
// lst[n-1]}

#include "haar.cuh"


using namespace std;

__device__ __constant__ stage_t deviceStages[FEATURENUM];
__device__ __constant__ stageMeta_t deviceStagesMeta[STAGENUM];


//-------------------------------------------------------------------------
// Simple Haar Cascade Kernel (single block, no shared memory)
//-------------------------------------------------------------------------
__global__ void allWindowsHaarCascade(float* deviceIntegralImage, int width, int height, float scaleStart)
{

}

//int SimpleCudaHaarCascade(unsigned char* outputImage, const float* integralImage, int width, int height)
//{
//	float scaleWidth = ((float)width) / 20.0f;
//	float scaleHeight = ((float)height) / 20.0f;
//	//int step;
//	//float scale;
//	cudaError_t cudaStatus;
//	float *deviceIntegralImage;
//
//	float scaleStart = scaleHeight < scaleWidth ? scaleHeight : scaleWidth;
//
//	int scaleMaxItt = (int)ceil(log(1 / scaleStart) / log(1.0 / 1.2));
//
//	//cout << scaleMaxItt << endl;
//
//	cudaStatus = cudaSetDevice(0);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//	}
//
//	//allocate GPU memory
//	cudaStatus = cudaMalloc((void**)&deviceIntegralImage, height * width * sizeof(float));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc of integralImage failed!");
//	}
//
//	// copy data to GPU memory
//	cudaStatus = cudaMemcpyToSymbol(deviceStagesMeta, stagesMeta, stageNum * sizeof(stageMeta_t), 0, cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy of StageMeta failed!");
//	}
//
//	cudaStatus = cudaMemcpyToSymbol(deviceStages, stagesFlat, featureNum * sizeof(stage_t), 0, cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy of StageMeta failed!");
//	}
//
//	cudaStatus = cudaMemcpy(deviceIntegralImage, integralImage, height * width * sizeof(float), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy of integralImage failed!");
//	}
//
//	//for (int sIdx = 0; sIdx < scaleMaxItt; ++sIdx)
//
//	//scale = scaleStart*(float)powf(1.0f / 1.2f, (float)(sIdx));
//	//cout << "Scale: " << scale << endl;
//
//	//step = (int)scale > 2 ? (int)scale : 2;
//
//	//int winWidth = (int)(20 * scale);
//	//int winHeight = (int)(20 * scale);
//
//
//	dim3 DimGridSimple(1, scaleMaxItt, 1);
//	dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
//
//	simpleCudaHaar << <DimGridSimple, DimBlock >> >(deviceIntegralImage, width, height, winWidth, winHeight, scale, step);
//
//	// Check for any errors launching the kernel
//	cudaStatus = cudaGetLastError();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "simpleCudaHaar launch failed: %s\n", cudaGetErrorString(cudaStatus));
//	}
//
//	// cudaDeviceSynchronize waits for the kernel to finish, and returns
//	// any errors encountered during the launch.
//	cudaStatus = cudaDeviceSynchronize();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching simpleCudaHaar!\n", cudaStatus);
//	}
//
//	cudaFree(deviceIntegralImage);
//
//	return 0;
//}


//-------------------------------------------------------------------------
// Simple Haar Cascade Kernel (single block, no shared memory)
//-------------------------------------------------------------------------
__global__ void simpleCudaHaar(float* deviceIntegralImage, int width, int height, int winWidth, int winHeight, float scale, int step)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int originX;
	int originY;
	int stageIndex;

	originX = (bx*BLOCK_SIZE+tx)*step;
	originY = (by*BLOCK_SIZE+ty)*step;

	// for each stage in stagesMeta
	feature_t *feature;
	float third = 0;
	if (originX <= (width - winWidth) && originY <= (height - winHeight)) {
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

int CudaHaarCascade(unsigned char* outputImage, const float* integralImage, int width, int height)
{
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

		dim3 DimGridSimple((width - winWidth) / (step * 32) + 1, (height - winHeight) / (step * 32) + 1, 1);
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
				printf("%f %d %d testing window at %d %d %d %d %d %d\n", scale, i, j, x, y, bx, by, tx, ty);
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