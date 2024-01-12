#include "cuda.h"
#include "cuda_runtime.h"
#include "hpc_helpers.hpp"
#include <stdio.h>

#define N 32 * 1024 * 1024
#define BLOCK_SIZE 256
#define WARP_SIZE 32

int getNumBlocks(int64_t n)
{
    int dev;
    cudaGetDevice(&dev);
    CUERR
    int sm_count;//5
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    CUERR
    int tpm;//2048
    cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    CUERR

    int num_blocks = std::max<int>(1, std::min<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                                                    sm_count * tpm / BLOCK_SIZE));
    return num_blocks;
}

template <typename T>
__device__ __forceinline__ T warpReduceSum(T val)
{
#pragma unroll
    for (int delta = WARP_SIZE>>1; delta > 0; delta >>=1)
    {
        val += __shfl_down_sync(0xffffffff, delta, val);
    }
    return val;
}
template <int NUM_PER_THREAD>
__global__ void myReduce(float *g_idata, float *g_odata, unsigned int n)
{
    float sum = 0;
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * NUM_PER_THREAD) + tid;

#pragma unroll
    for (int gap = 0; gap < NUM_PER_THREAD; gap++)
    {
        sum += g_idata[i + gap * blockDim.x];
    }
    __shared__ float warpLevelSums[WARP_SIZE];
    int const laneId = threadIdx.x / warpSize;
    int const warpId = threadIdx.x / warpSize;
}

int main()
{
    const int block_num = getNumBlocks(N);
    const int NUM_PER_BLOCK = N / block_num;
    const int NUM_PER_THREAD = NUM_PER_BLOCK / BLOCK_SIZE;
}