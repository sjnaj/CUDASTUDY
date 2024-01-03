#include <cuda.h>
#include "common/book.h"
#define N (33 * 1024)

__global__ void add(int *a, int *b, int *c)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) // 一个线程处理对应一列，不限长度
    {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main(void)
{
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    HANDLE_ERROR(cudaMalloc((void **)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_c, N * sizeof(int)));

    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i * i;
    }

    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

    add<<<128, 128>>>(dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    bool sucess = true;

    for (int i = 0; i < N; i++)
    {
        if ((a[i] + b[i]) != c[i])
        {
            printf("error\n");
            sucess = false;
        }
    }
    if (sucess)
    {
        printf("okk\n");
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}