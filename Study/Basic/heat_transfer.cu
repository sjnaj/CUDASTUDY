#include <cuda.h>
#include "parallelprogrammingbook/include/hpc_helpers.hpp"
#include "common/cpu_bitmap.h"
#include "common/cpu_anim.h"
#include "common/book.h"
#define DIM 1024
#define MAX_TEMP 0.8f
#define MIN_TEMP 0.0001f
#define SPEED 0.25f

/*heating cells remains at a constant temperature*/
__global__ void copy_const_kernel(float *iptr, float *const cptr)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    if (cptr[offset] != 0)
    {
        iptr[offset] = cptr[offset];
    }
}

__global__ void blend_kernel(float *outSrc, float *const inSrc)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int left = offset - 1;
    int right = offset + 1;
    if (x == 0)
        left++;
    if (x == DIM - 1)
        right--;

    int top = offset - DIM;
    int bottom = offset + DIM;
    if (y == 0)
        top += DIM;
    if (y == DIM - 1)
        bottom -= DIM;

    outSrc[offset] = inSrc[offset] + SPEED * (inSrc[top] + inSrc[bottom] + inSrc[left] + inSrc[right] - 4 * inSrc[offset]);
}

struct DataBlock
{
    unsigned char *output_bitmap;
    float *dev_inSrc;
    float *dev_outSrc;
    float *dev_constSrc;
    CPUAnimBitmap *bitmap;
    cudaEvent_t start, stop;
    float totalTime;
    float frames;
};

void anim_gpu(DataBlock *d, int ticks)
{
    cudaEventRecord(d->start, 0);
    CUERR
    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    CPUAnimBitmap *bitmap = d->bitmap;

    for (int i = 0; i < 90; i++)
    {
        copy_const_kernel<<<blocks, threads>>>(d->dev_inSrc, d->dev_constSrc);
        blend_kernel<<<blocks, threads>>>(d->dev_outSrc, d->dev_inSrc);
        swap(d->dev_inSrc, d->dev_outSrc);
    }

    float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_inSrc);

    cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);
    CUERR
    cudaEventRecord(d->stop, 0);
    CUERR
    cudaEventSynchronize(d->stop);
    CUERR
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, d->start, d->stop);
    CUERR
    d->totalTime += elapsedTime;

    d->frames++;
    printf("average time per frame: %3.1f ms\n", d->totalTime / d->frames);
}

void anim_exit(DataBlock *d)
{
    cudaFree(d->dev_inSrc);
    CUERR
    cudaFree(d->dev_outSrc);
    CUERR
    cudaFree(d->dev_constSrc);
    CUERR

    cudaEventDestroy(d->start);
    CUERR
    cudaEventDestroy(d->stop);
    CUERR
}

int main(void)
{

    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;

    cudaEventCreate(&data.start);
    CUERR
    cudaEventCreate(&data.stop);
    CUERR
    cudaMalloc((void **)&data.output_bitmap, bitmap.image_size());
    CUERR
    cudaMalloc((void **)&data.dev_inSrc, bitmap.image_size());
    CUERR
    cudaMalloc((void **)&data.dev_outSrc, bitmap.image_size());
    CUERR
    cudaMalloc((void **)&data.dev_constSrc, bitmap.image_size());
    CUERR
    float *temp;
    cudaMallocHost((void **)&temp, bitmap.image_size());
    CUERR

    // 初始化内部热点
    for (int i = 0; i < DIM * DIM; i++)
    {
        temp[i] = 0;
        int x = i % DIM;
        int y = i / DIM;
        if ((x > 300) && (x < 600) && (y > 310) && (y < 601))
        {
            temp[i] = MAX_TEMP;
        }
    }
    temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
    temp[DIM * 700 + 100] = MIN_TEMP;
    temp[DIM * 300 + 300] = MIN_TEMP;
    temp[DIM * 200 + 700] = MIN_TEMP;
    for (int y = 800; y < 900; y++)
    {
        for (int x = 400; x < 500; x++)
        {
            temp[x + y * DIM] = MIN_TEMP;
        }
    }

    cudaMemcpy(data.dev_constSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice);
    CUERR
    // 初始化外部影响
    for (int y = 800; y < DIM; y++)
    {
        for (int x = 0; x < 200; x++)
        {
            temp[x + y * DIM] = MAX_TEMP;
        }
    }
    cudaMemcpy(data.dev_inSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice);
    CUERR

    cudaFreeHost(temp);
    CUERR

    bitmap.anim_and_exit((void (*)(void *, int))anim_gpu, (void (*)(void *))anim_exit);
}
