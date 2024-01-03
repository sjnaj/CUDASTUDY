#include <cuda.h>
#include "common/book.h"
#include "common/cpu_bitmap.h"

#define rnd(x) (x * rand() / RAND_MAX)
#define SPHERES 50
#define INF 2e10f
#define DIM 1024
struct Sphere
{
    float r, b, g;
    float radius;
    float x, y, z;
    __device__ float hit(float ox, float oy, float *n)
    { // 射线向z轴正方向射出
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy < radius * radius)
        {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf(radius * radius); // 交点与球心水平距离的度
            return  z-dz;                    // 与交点的竖直距离
        }
        return INF+1;
    }
};
__constant__ Sphere s[SPHERES];

__global__ void kernel(unsigned char *ptr)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float ox = (x - DIM / 2);
    float oy = (y - DIM / 2);

    float r = 0, g = 0, b = 0;
    float maxz = INF;

    for (int i = 0; i < SPHERES; i++)
    {
        float fscale;
        float t = s[i].hit(ox, oy, &fscale);
        if (t <maxz)
        {
             maxz=t;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
        }
    }
    ptr[offset * 4 + 0] = (int)(r * 255);
    ptr[offset * 4 + 1] = (int)(g * 255);
    ptr[offset * 4 + 2] = (int)(b * 255);
    ptr[offset * 4 + 3] = 255;
}

int main(void)
{
    cudaEvent_t start,stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start,0));

    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;

    HANDLE_ERROR(cudaMalloc((void **)&dev_bitmap, bitmap.image_size()));

    Sphere *temp_s = (Sphere *)malloc(sizeof(Sphere) * SPHERES);
srand((unsigned int)time(NULL));

    for (int i = 0; i < SPHERES; i++)
    {
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);

        temp_s[i].x = rnd((float)DIM) - DIM / 2.0f;
        temp_s[i].y = rnd((float)DIM) - DIM / 2.0f;
        temp_s[i].z = rnd((float)DIM)+DIM / 10.f ;
        temp_s[i].radius = rnd(DIM / 10.f);
    }

    HANDLE_ERROR(cudaMemcpyToSymbol(s, temp_s, sizeof(Sphere) * SPHERES));
    free(temp_s);

    dim3 grids(DIM / 16, DIM / 16);
    dim3 threads(16, 16);

    kernel<<<grids, threads>>>(dev_bitmap);

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaEventRecord(stop,0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,start,stop));
    printf("Time to generate: %.3f ms\n",elapsedTime);
    bitmap.display_and_exit();
    cudaFree(dev_bitmap);
}