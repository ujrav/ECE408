// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... +
// lst[n-1]}

#include "integralImg.cuh"

using namespace std;


void transposeSerial(float *input, float *output, int width, int height)
{
	for (int r = 0; r < height; ++r)
	{
		for (int c = 0; c < width; ++c)
		{
//			output[c][r] = input[r][c];
		}
	}
}

//-------------------------------------------------------------------------
// Parallel Scan Add Auxiliary Kernel (Part 3)
//-------------------------------------------------------------------------
__global__ void scanAddAux(float *aux, float *output, int len)
{

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
__global__ void scanRow(float *input, float *output, float *aux, int len)
{
	// LECTURE METHOD with 2 reads and 2 writes
	__shared__ float scanBlock[SCAN_BLOCK_SIZE << 1];

	int tx = threadIdx.x;
	int bdx = blockDim.x;
	int biy = blockIdx.y;
	int bix = blockIdx.x;
	int start = 2 * bdx * bix;
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
		aux[bix] = scanBlock[2 * SCAN_BLOCK_SIZE - 1];
	}

}


int CudaIntegralImage(float* grayImage, float* integralImage, int width, int height)
{
	cudaError_t cudaStatus;
	float *deviceInputImage;
	float *deviceIntegralImage;
	float *deviceAuxInput;
	//float *deviceAuxOutput;
	//float *deviceTranspose;
	float *hostTranspose;
	int numElements; // number of elements in the list
	int gridCols, gridRows;	// grid dimensions

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	numElements = width;
	gridCols = (int)ceil(float(numElements) / float(SCAN_BLOCK_SIZE << 1));
	gridRows = height;

	hostTranspose = (float *)malloc(width * height * sizeof(float));

	//allocate GPU memory
	cudaStatus = cudaMalloc((void **)&deviceInputImage, width * height * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc of deviceInputImage failed!");
	}

	cudaStatus = cudaMalloc((void **)&deviceIntegralImage, width * height * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc of deviceInputImage failed!");
	}

	cudaStatus = cudaMalloc((void **)&deviceAuxInput, gridCols * gridRows * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc of deviceInputImage failed!");
	}

	//cudaStatus = cudaMalloc((void **)&deviceAuxOutput, gridCols * gridRows * sizeof(float));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc of deviceInputImage failed!");
	//}

	//cudaStatus = cudaMalloc((void **)&deviceTranspose, width * height * sizeof(float));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc of deviceInputImage failed!");
	//}

	cudaStatus = cudaMemcpy(deviceInputImage, grayImage, width * height * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy of inputImage failed!");
	}

	dim3 DimGrid(gridCols, gridRows, 1);
	dim3 DimBlock(SCAN_BLOCK_SIZE, 1, 1);
	dim3 DimGrid2(1, 1, 1);


	cudaMemset(deviceAuxInput, 0, gridCols * sizeof(float));
	//cudaMemset(deviceAuxOutput, 0, gridCols * sizeof(float));
	printf("Kernel with dimensions %d x %d x %d launching\n", DimGrid.x, DimGrid.y, DimGrid.z);

	scanRow << < DimGrid, DimBlock >> > (deviceInputImage, deviceIntegralImage, deviceAuxInput, numElements);	// Part 1
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching scanRow!\n", cudaStatus);
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "scanRow launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaMemcpy(integralImage, deviceIntegralImage, width * height * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy of integralImage failed!");
	}

	transposeSerial(integralImage, hostTranspose, width, height);

	FILE *fp;

	fp = fopen("integralParallel.txt", "w");
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			fprintf(fp, "%f ", integralImage[j + i*width]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);


	//cudaStatus = cudaMemcpy(deviceTranspose, hostTranspose, width * height * sizeof(float), cudaMemcpyHostToDevice);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy of inputImage failed!");
	//}

	//gridCols = (int)ceil(float(height) / float(SCAN_BLOCK_SIZE << 1));
	//gridRows = width;

	//dim3 DimGridT(gridCols, gridRows, 1);
	//dim3 DimBlockT(SCAN_BLOCK_SIZE, 1, 1);

	//scanRow << < DimGridT, DimBlockT >> > (deviceTranspose, deviceIntegralImage, deviceAuxInput, gridRows);	// Part 1
	//// cudaDeviceSynchronize waits for the kernel to finish, and returns
	//// any errors encountered during the launch.
	//cudaStatus = cudaDeviceSynchronize();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching scanRow!\n", cudaStatus);
	//}
	//////scanRow << < DimGrid2, DimBlock >> > (deviceAuxInput, deviceAuxOutput, NULL, gridCols);					// Part 2
	//////cudaDeviceSynchronize();
	//////scanAddAux << < DimGrid, DimBlock >> > (deviceAuxOutput, deviceIntegralImage, numElements);				// Part 3
	//////cudaDeviceSynchronize();



	//cudaStatus = cudaMemcpy(hostTranspose, deviceIntegralImage, width * height * sizeof(float), cudaMemcpyDeviceToHost);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy of hostTranspose failed!");
	//}

	//transposeSerial(hostTranspose, integralImage, width, height);








	//cudaFree(deviceAuxInput);
	//cudaFree(deviceAuxOutput);
	//cudaFree(deviceIntegralImage);
	//cudaFree(deviceInputImage);
	//free(hostTranspose);

	return 0;
}