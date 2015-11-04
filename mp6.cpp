// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define TILE_WIDTH 1024
#define BLOCK_SIZE 1024

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)                                                     

__global__ void printCDF(float *input, int len){
  if (threadIdx.x == 0){
    for (int i = 0; i < len; i++){
      // printf("%d %f\n", i, input[i]);
    }
  }
}

__global__ void printHisto(int *input, int len){
  if (threadIdx.x == 0){
    for (int i = 0; i < len; i++){
      // printf("%d %d\n", i, input[i]);
    }
  }
}

__global__ void printBoth(int *input, float *input2, int len){
  if (threadIdx.x == 0){
    for (int i = 0; i < len; i++){
      // printf("%d %d %f\n", i, input[i], input2[i]);
    }
  }
}
__global__ void scanFunc(int *input, float *output, int len){
  __shared__ float buf[2*BLOCK_SIZE];
  int bx, tx, index, stride;
  tx = threadIdx.x;
  bx = blockIdx.x;
  index = bx*(2*BLOCK_SIZE) + tx;

  if (index < len){
    buf[tx] = input[index] / ((float) (len));
  }
  else{
    buf[tx] = 0;
  }

  index = index + BLOCK_SIZE;
  if (index < len){
    buf[tx + BLOCK_SIZE] = input[index] / ((float) (len));
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
}

__global__ void hist(float *input, unsigned char* uchar, int *histo, int width, int height) {
  __shared__ int histo_private[256];
  int tx, ty, bx, by;
  int index, idx;
  unsigned char r, b, g;
  unsigned char gray;
  tx = threadIdx.x;
  ty = threadIdx.y;
  bx = blockIdx.x;
  by = blockIdx.y;
  index = bx*TILE_WIDTH + tx;
  idx = tx;

  if (idx < 256){
    histo_private[idx] = 0;
  }

  __syncthreads();

  if (index < width * height){
    uchar[3 * index] = r = (unsigned char) (255 * input[3 * index]);
    uchar[3 * index + 1] = b = (unsigned char) (255 * input[3 * index + 1]);
    uchar[3 * index + 2] = g = (unsigned char) (255 * input[3 * index + 2]);

    gray = (unsigned char) (0.21*r + 0.71*g + 0.07*b);

    //if (index < 100 || width * height - index < 100)
      // printf("%d %d %d %d\n", index, uchar[3 * index], uchar[3 * index + 1], uchar[3 * index + 2]);

    atomicAdd(&(histo_private[gray]), 1);
  }

  __syncthreads();

  if (idx < 256){
    atomicAdd(&(histo[idx]), histo_private[idx]);
  }
}

__global__ void correct(unsigned char *uchar, float *output, float *cdf, int len){
  int tx, bx, index;
  tx = threadIdx.x;
  bx = blockIdx.x;
  float correctVal;
  float correctValSave;
  unsigned char val;
  float cdfmin = cdf[0];

  index = bx * TILE_WIDTH + tx;
  if (index < len){
    // if (index == 3*(85 * 256 + 85) + 1){
    //   printf("lolololo\n");
    // }
    val = uchar[index];
    correctVal = 255.0*(cdf[val] - cdfmin)/(1.0 - cdfmin);
    correctValSave = correctVal;
    if (correctVal < 0){
      correctVal = 0;
    }

    if (correctVal > 255){
      correctVal = 255;
    }

    output[index] = (float)(correctVal / 255.0);
    // if (index == 3*(85 * 256 + 85) + 1){
    //   printf("lalalala\n");
    //   output[index] = .976;
    // }
  }
}
//@@ insert code here

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;
  float *deviceInputImageData;
  unsigned char *deviceImageChar;
  float *deviceOutputImageData;
  int *deviceHisto;
  float *deviceCDF;
  int empty[256] = {0};

  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  hostOutputImageData = (float *) malloc(imageWidth * imageHeight * 3 * sizeof (float));
  hostInputImageData = wbImage_getData(inputImage);

  wbCheck(cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * 3 * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceImageChar, imageWidth * imageHeight * 3 * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * 3 * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceHisto, 256 * sizeof(int)));
  wbCheck(cudaMalloc((void **)&deviceCDF, 256 * sizeof(int)));

  wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * 3 * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceHisto, empty, 256 * sizeof(int),
                     cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceCDF, empty, 256 * sizeof(int),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid((imageWidth * imageHeight)/TILE_WIDTH, 1, 1);
  if ((imageWidth * imageHeight)%TILE_WIDTH) DimGrid.x++;
  dim3 DimBlock(TILE_WIDTH, 1, 1);

  dim3 DimGridCorrect((imageWidth * imageHeight * 3)/TILE_WIDTH, 1, 1);
  if ((imageWidth * imageHeight * 3)%TILE_WIDTH) DimGridCorrect.x++;

  dim3 DimGridS(1, 1, 1);
  dim3 DimBlockS(256, 1, 1);

  wbLog(TRACE, "image width = ", imageWidth, " height = ", imageHeight);
  wbLog(TRACE, "grid dimensions x = ", DimGrid.x, " y = ", DimGrid.y, " z = ", DimGrid.z);
  wbLog(TRACE, "grid Correct dimensions x = ", DimGridCorrect.x, " y = ", DimGridCorrect.y, " z = ", DimGridCorrect.z);

  wbTime_start(Compute, "Performing CUDA histogram");

  hist<<<DimGrid, DimBlock>>>(deviceInputImageData, deviceImageChar, deviceHisto, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  wbTime_stop(Compute, "Performing CUDA histogram");

  wbTime_start(Compute, "Performing CUDA CDF");

  scanFunc<<<DimGridS, DimBlockS>>>(deviceHisto, deviceCDF, imageWidth * imageHeight);
  cudaDeviceSynchronize();

  wbTime_stop(Compute, "Performing CUDA CDF");

  wbTime_start(Compute, "Performing CUDA correct");

  correct<<<DimGridCorrect, DimBlock>>>(deviceImageChar, deviceOutputImageData, deviceCDF, imageWidth * imageHeight * 3);
  cudaDeviceSynchronize();

  wbTime_stop(Compute, "Performing CUDA correct");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * 3 * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbImage_setData(outputImage, hostOutputImageData);
  wbSolution(args, outputImage);

  //@@ insert code here

  return 0;
}
