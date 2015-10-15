#include    <wb.h>

#define wbCheck(stmt) do {                                         \
        cudaError_t err = stmt;                                    \
        if (err != cudaSuccess) {                                  \
            wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err)); \
            wbLog(ERROR, "Failed to run stmt ", #stmt);            \
            return -1;                                             \
        }                                                          \
    } while(0)


//@@ Define any useful program-wide constants here
#define KERNEL_WIDTH 3
#define KERNEL_RADIUS 1
#define TILE_WIDTH 8

//@@ Define constant memory for device kernel here
__device__ __constant__ float mask[KERNEL_WIDTH][KERNEL_WIDTH][KERNEL_WIDTH];

__device__ __constant__ float test[3][3][3];

__global__ void conv3d(float *A, float *B,
                       const int z_size, const int y_size, const int x_size) {
    //@@ Insert kernel code here
    __shared__ float buf[TILE_WIDTH + 2][TILE_WIDTH + 2][TILE_WIDTH + 2];
    int tx, ty, tz;
    int bx, by, bz;
    int z_in, x_in, y_in;
    int i, j, k;
    float val = 0;
    int x, y, z;

    tx = threadIdx.x;
    ty = threadIdx.y;
    tz = threadIdx.z;
    bx = blockIdx.x;
    by = blockIdx.y;
    bz = blockIdx.z;

    z_in = bz * TILE_WIDTH + tz;
    y_in = by * TILE_WIDTH + ty;
    x_in = bx * TILE_WIDTH + tx;

    x = tx + 1;
    y = ty + 1;
    z = tz + 1;

    if (tz == TILE_WIDTH + KERNEL_RADIUS)
    {
        z_in = z_in - TILE_WIDTH - 2;
        z = 0;
    }
    if (ty == TILE_WIDTH + KERNEL_RADIUS)
    {
        y_in = y_in - TILE_WIDTH - 2;
        y = 0;
    }
    if (tx == TILE_WIDTH + KERNEL_RADIUS)
    {
        x_in = x_in - TILE_WIDTH - 2;
        x = 0;
    }

    if (z_in < 0 || z_in >= z_size || y_in < 0 || y_in >= y_size || x_in < 0 || x_in >= x_size)
    {
        buf[z][y][x] = 0;
    }
    else
    {
       
        //printf("%d\n", z_in*(y_size * x_size) + y_in*(x_size) + x_in);
        buf[z][y][x] = A[z_in*(y_size * x_size) + y_in*(x_size) + x_in];
    }
    __syncthreads();

    if (tx < TILE_WIDTH && ty < TILE_WIDTH && tz < TILE_WIDTH && z_in < z_size && y_in < y_size && x_in < x_size)
    {
        for (i = 0; i < KERNEL_WIDTH; i++)
        {
            for (j = 0; j < KERNEL_WIDTH; j++)
            {
                for (k = 0; k < KERNEL_WIDTH; k++)
                {

                    val += mask[k][j][i] * buf[tz + k][ty + j][tx + i];

                }
            }
        }
        B[z_in*(y_size * x_size) + y_in*(x_size) + x_in] = val;
    }
    
}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int z_size;
    int y_size;
    int x_size;
    int inputLength, kernelLength;
    float * hostInput;
    float * hostKernel;
    float * hostOutput;
    float * deviceInput;
    float * deviceOutput;
    float testvar[27] = {0};

    testvar[0] = 323454;
    testvar[1] = 987654567;

    args = wbArg_read(argc, argv);

    // Import data
    hostInput = (float*) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostKernel = (float*) wbImport(wbArg_getInputFile(args, 1), &kernelLength);
    hostOutput = (float*) malloc(inputLength * sizeof(float));

    // First three elements are the input dimensions  
    z_size = hostInput[0];
    y_size = hostInput[1];
    x_size = hostInput[2];
    wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
    wbLog(TRACE, "inputLength ", inputLength);
    assert(z_size * y_size * x_size == inputLength - 3);
    assert(kernelLength == 27);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    //@@ Allocate GPU memory here
    // Recall that inputLength is 3 elements longer than the input data
    // because the first  three elements were the dimensions
    wbCheck(cudaMalloc( (void**) &deviceInput, (inputLength - 3)* sizeof(float)));
    wbCheck(cudaMalloc( (void**) &deviceOutput, (inputLength - 3)* sizeof(float)));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    //@@ Copy input and kernel to GPU here
    // Recall that the first three elements of hostInput are dimensions and do
    // not need to be copied to the gpu
    wbCheck(cudaMemcpyToSymbol(test, hostKernel, 27*sizeof(float), 0, cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpyToSymbol(mask, hostKernel, (27) * sizeof(float), 0, cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceInput, &(hostInput[3]), (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice));
    
    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ Initialize grid and block dimensions here
    dim3 DimGrid(x_size/TILE_WIDTH, y_size/TILE_WIDTH, z_size/TILE_WIDTH);
    if (x_size%TILE_WIDTH) DimGrid.x++;
    if (y_size%TILE_WIDTH) DimGrid.y++;
    if (z_size%TILE_WIDTH) DimGrid.z++;
    dim3 DimBlock(TILE_WIDTH + KERNEL_WIDTH - 1, TILE_WIDTH + KERNEL_WIDTH - 1, TILE_WIDTH + KERNEL_WIDTH - 1);

    wbLog(TRACE, "grid dimensions z = ", DimGrid.z, " y = ", DimGrid.y, " x = ", DimGrid.x);
    //@@ Launch the GPU kernel here
    conv3d<<<DimGrid,DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    //@@ Copy the device memory back to the host here
    // Recall that the first three elements of the output are the dimensions
    // and should not be set here (they are set below)
    wbCheck(cudaMemcpy(&(hostOutput[3]), deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    // Set the output dimensions for correctness checking
    hostOutput[0] = z_size;
    hostOutput[1] = y_size;
    hostOutput[2] = x_size;
    wbSolution(args, hostOutput, inputLength);

    // Free device memory
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    // Free host memory
    free(hostInput);
    free(hostOutput);
    return 0;
}
