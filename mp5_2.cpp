// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this
#define MAX_GRID 65536

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)


__device__ float S[MAX_GRID];

__global__ void scanFunc(float *input, float *output, int len){
  __shared__ float buf[2*BLOCK_SIZE];
  int bx, tx, index, stride;
  tx = threadIdx.x;
  bx = blockIdx.x;
  index = bx*(2*BLOCK_SIZE) + tx;

  if (index < len){
    buf[tx] = input[index];
  }
  else{
    buf[tx] = 0;
  }

  index = index + BLOCK_SIZE;
  if (index < len){
    buf[tx + BLOCK_SIZE] = input[index];
  }
  else{
    buf[tx + BLOCK_SIZE] = 0;
  }

  __syncthreads();

  for (stride = 1; stride <= BLOCK_SIZE; stride *= 2){
    index = (tx + 1) * 2 * stride - 1;
    if (index < 2*BLOCK_SIZE){
      buf[index] += buf[index-stride];
    }
    __syncthreads();
  }

  for (stride = BLOCK_SIZE/2; stride > 0; stride /= 2){
    __syncthreads();
    index = (tx + 1) * 2 * stride - 1;
    if (index + stride < 2 * BLOCK_SIZE){
      buf[index + stride] += buf[index];
    }
  }

  __syncthreads();

  index = bx*(2*BLOCK_SIZE) + tx;

  if (index < len){
    output[index] = buf[tx];
  }

  index = index + BLOCK_SIZE;
  if (index < len){
    output[index] = buf[tx + BLOCK_SIZE];
  }

  if ((tx == BLOCK_SIZE - 1 && index < len) || (index == len - 1)){
    S[bx] = buf[tx + BLOCK_SIZE];
  }
}

__global__ void scanGlobal(int len){
  __shared__ float buf[2*BLOCK_SIZE];
  int bx, tx, index, stride;
  tx = threadIdx.x;
  bx = blockIdx.x;
  index = bx*(2*BLOCK_SIZE) + tx;

  if (index < len){
    buf[tx] = S[index];
  }
  else{
    buf[tx] = 0;
  }

  index = index + BLOCK_SIZE;
  if (index < len){
    buf[tx + BLOCK_SIZE] = S[index];
  }
  else{
    buf[tx + BLOCK_SIZE] = 0;
  }

  __syncthreads();

  for (stride = 1; stride <= BLOCK_SIZE; stride *= 2){
    index = (tx + 1) * 2 * stride - 1;
    if (index < 2*BLOCK_SIZE){
      buf[index] += buf[index-stride];
    }
    __syncthreads();
  }

  for (stride = BLOCK_SIZE/2; stride > 0; stride /= 2){
    __syncthreads();
    index = (tx + 1) * 2 * stride - 1;
    if (index + stride < 2 * BLOCK_SIZE){
      buf[index + stride] += buf[index];
    }
  }

  __syncthreads();

  index = bx*(2*BLOCK_SIZE) + tx;

  if (index < len){
    S[index] = buf[tx];
  }

  index = index + BLOCK_SIZE;
  if (index < len){
    S[index] = buf[tx + BLOCK_SIZE];
  }
}

__global__ void scanAdd(float *output, int len){
  int bx, tx, index;
  tx = threadIdx.x;
  bx = blockIdx.x;
  index = bx*(2*BLOCK_SIZE) + tx;

  if (index < len && bx > 0){
    output[index] += S[bx - 1];
  }

  index = index + BLOCK_SIZE;
  if (index < len && bx > 0){
    output[index] += S[bx - 1];
  }
}
__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from here
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ", numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(numElements/(BLOCK_SIZE << 1), 1, 1);
  if (numElements%(BLOCK_SIZE << 1)) DimGrid.x++;
  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  wbLog(TRACE, "grid dimensions z = ", DimGrid.z, " y = ", DimGrid.y, " x = ", DimGrid.x);

  dim3 DimSGrid(DimGrid.x/(BLOCK_SIZE << 1), 1, 1);
  if (DimGrid.x%(BLOCK_SIZE << 1)) DimSGrid.x++;

  wbLog(TRACE, "S grid dimensions z = ", DimSGrid.z, " y = ", DimSGrid.y, " x = ", DimSGrid.x);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scanFunc<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numElements);
  cudaDeviceSynchronize();

  scanGlobal<<<DimSGrid, DimBlock>>>(numElements/(BLOCK_SIZE << 1));
  cudaDeviceSynchronize();

  scanAdd<<<DimGrid, DimBlock>>>(deviceOutput, numElements);
  cudaDeviceSynchronize();

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
