#include <cuda.h>
#include <stdio.h>
#include <vector>
#include <numeric>
#include "/home/fengsc/CUDASTUDY/Study/include/CudaAllocator.h"
#define WARP_SIZE 32
#define N 10000
#define BLOCK_SIZE 128

__device__ __forceinline__ float warp_reduce_sum(float val)
{
#pragma unroll
    for (int delta = WARP_SIZE >> 1; delta > 0; delta >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, delta);
    }
    return val;
}

__global__ void reduce_sum(float * __restrict__ in, float * __restrict__ res, int n)
{
    int tid = threadIdx.x;
    int id = blockDim.x * blockIdx.x + tid;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    constexpr int num_warps = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    static __shared__ float smem[num_warps];
    float sum = id < n ? in[id] : 0.0f;
    sum = warp_reduce_sum(sum);
    if (lane_id == 0)
        smem[warp_id] = sum;
    __syncthreads();
    sum = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
    if (warp_id == 0)
    {
        sum = warp_reduce_sum(sum);
    }
    if (tid == 0)
    {
        atomicAdd(res, sum);
    }
}
int main()
{
    std::vector<float, CudaAllocator<float>> in(N, 1, {});
    std::vector<float, CudaAllocator<float>> res(1, 0, {});
    float sum = 0;
    for (auto &i : in)
    {
        i = std::rand() % 4;
        sum += i;
    }
    printf("%f", sum);
    dim3 Grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    dim3 Block(BLOCK_SIZE, 1);
    reduce_sum<<<Grid, Block>>>(in.data(), res.data(), N);
    cudaDeviceSynchronize();
    printf("%f", res[0]);
}