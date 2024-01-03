#include "common/book.h"
#include <cuda.h>
#define imin(a, b) (a < b ? a : b)

int const N = 1024*33;
int const threadsPreBlock = 256;//是2的指数才能被规约
int const blocksPerGrid = imin(32, (N + threadsPreBlock - 1) / threadsPreBlock);

__global__ void dot(float *a, float *b, float *c)
{
   
    __shared__ float cache[threadsPreBlock];//块内共享，类似static
     
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < N)
    {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
        
    }
   
    cache[cacheIndex] = temp;

    __syncthreads();

    int i = blockDim.x / 2;

    while (i != 0)
    {
        if (cacheIndex < i)
        {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads(); // 块内同步，不进行reduction的线程也要同步，否则会被等待执行。
        i /= 2;
    }
    if (cacheIndex == 0)
    {
        c[blockIdx.x] = cache[0];
    }
}

int main(void)
{
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    a = (float *)malloc(N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));
    partial_c = (float *)malloc(blocksPerGrid * sizeof(float));

    HANDLE_ERROR(cudaMalloc((void **)&dev_a, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_b, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_partial_c, blocksPerGrid*sizeof(float)));

    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    HANDLE_ERROR(cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice));
    dot<<<blocksPerGrid, threadsPreBlock>>>(dev_a, dev_b, dev_partial_c);

    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost));

    c = 0;
    for (int i = 0; i < blocksPerGrid; i++)
    {
        c += partial_c[i];
    }

#define sum_square(x) (x * (x + 1) * (2 * x + 1) / 6)

    printf("Does Gpu value %.6g=%.6g?\n", c, 2 * sum_square((float)(N - 1)));
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    free(a);
    free(b);
    free(partial_c);
    
}