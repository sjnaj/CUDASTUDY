#include <cuda.h>
#include "parallelprogrammingbook/include/hpc_helpers.hpp"
#include "common/cpu_bitmap.h"
#include "common/cpu_anim.h"
#include "common/book.h"
#define DIM 1024
#define MAX_TEMP 0.8f
#define MIN_TEMP 0.0001f
#define SPEED 0.25f

//纹理引用被弃用，效果不如访问global变量 

// 需要全局声明，不能作为函数参数传递
texture<float, 2> texConstSrc;
texture<float, 2> texIn;
texture<float, 2> texOut;

/*heating cells remains at a constant temperature*/
__global__ void copy_const_kernel(float *iptr,cudaTextureObject_t const constSrc)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float c = tex2D<float>(constSrc, x, y);
    if (c != 0)
    {
        iptr[offset] = c;
    }
}

__global__ void blend_kernel(float *dst, bool dstOut)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float t, l, c, r, b;

    if (dstOut)
    {
        // 越界的会自动回退
        t = tex2D(texIn, x, y - 1);
        l = tex2D(texIn, x - 1, y);
        c = tex2D(texIn, x, y);
        r = tex2D(texIn, x + 1, y);
        b = tex2D(texIn, x, y + 1);
    }
    else
    {
        t = tex2D(texOut, x, y - 1);
        l = tex2D(texOut, x - 1, y);
        c = tex2D(texOut, x, y);
        r = tex2D(texOut, x + 1, y);
        b = tex2D(texOut, x, y + 1);
    }
    dst[offset] = c + SPEED * (t + l + r + b - 4 * c); // 通过纹理访问输入，通过global修改输出。
}

struct DataBlock
{
    unsigned char *output_bitmap;
    float *dev_inSrc;
    float *dev_outSrc;
    float *dev_constSrc;
    cudaTextureObject_t text_constSrc;
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

    volatile bool dstOut = true;

    for (int i = 0; i < 90; i++)
    {
        float *in, *out;
        if (dstOut)
        {
            in = d->dev_inSrc;
            out = d->dev_outSrc;
        }
        else
        {
            out = d->dev_inSrc;
            in = d->dev_outSrc;
        }
        copy_const_kernel<<<blocks, threads>>>(in,d->text_constSrc);
        blend_kernel<<<blocks, threads>>>(out, dstOut);
        dstOut = !dstOut;
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
    cudaUnbindTexture(texIn);
    CUERR
    cudaUnbindTexture(texOut);
    CUERR
    cudaUnbindTexture(texConstSrc);
    CUERR
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
    long const imageSize = bitmap.image_size();
    cudaEventCreate(&data.start);
    CUERR
    cudaEventCreate(&data.stop);
    CUERR
    cudaMalloc((void **)&data.output_bitmap, imageSize);
    CUERR
    cudaMalloc((void **)&data.dev_inSrc, imageSize);
    CUERR
    cudaMalloc((void **)&data.dev_outSrc, imageSize);
    CUERR
    cudaMalloc((void **)&data.dev_constSrc, imageSize);
    CUERR

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaBindTexture2D(NULL, texConstSrc, data.dev_constSrc, desc, DIM, DIM, sizeof(float) * DIM);
    CUERR
    cudaBindTexture2D(NULL, texIn, data.dev_inSrc, desc, DIM, DIM, sizeof(float) * DIM);
    CUERR
    cudaBindTexture2D(NULL, texOut, data.dev_outSrc, desc, DIM, DIM, sizeof(float) * DIM);
    CUERR

    float *temp;
    cudaMallocHost((void **)&temp, imageSize);
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

    // cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaArray *cuArray;
    cudaMallocArray(&cuArray, &desc, DIM * sizeof(float), DIM);
    cudaMemcpy2DToArray(cuArray, 0, 0, temp, DIM * sizeof(float), DIM * sizeof(float), DIM, cudaMemcpyHostToDevice);
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;//越界的设为边界值
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaCreateTextureObject(&data.text_constSrc, &resDesc, &texDesc, NULL);

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
