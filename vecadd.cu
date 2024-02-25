#define N 10000
#define BlockNum 32
#define BlockSize 128
float v1[N]
float v2[N]
float v3[N]

v1_dev
v2_dev

__global__ kernel(float* __restrict__v1,float* __restrict__ v2,float* __restrict__ v3,int len){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    
    int i,iter=0;
    #pragma unroll
    while((i=idx+iter*BlockSize)<len){
        v3[i]=v1[i]*v2[i];
        iter++;
    }
}


int main(){
    kernel<<<BlockNum,BlockSize>>>(v1,v2,v3,N);
}