#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "scatter_cuda_kernel.h"


__global__ void scatter_map_kernel(int* before_index, int* after_index, int* values, int length, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i < total){
        int values_index = i/length;
        int key_index = i%length;

        if(values[values_index]==before_index[key_index]){
            values[values_index] = after_index[key_index];
        }
        
    }

}

void scatter_map_cuda(int* before_index, int* after_index, int* values, int class_length, int values_length, cudaStream_t stream)
{
    cudaError_t err;

    int total = class_length*values_length;
    int block_size = 1024;
    int grid_size = total/block_size+1;

    scatter_map_kernel<<<grid_size, block_size, 0, stream>>>(before_index, after_index, values, class_length, total);

    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__global__ void scatter_sum_kernel(int* index, float* values, float* sum_result, int length, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i < total){
        int values_index = i/length;
        int key_index = i%length;

        if(index[values_index]==key_index){
            atomicAdd(&sum_result[key_index], values[values_index]);
        }
        
    }

}

void scatter_sum_cuda(int* index, float* values, float* sum_result, int class_length, int values_length, cudaStream_t stream)
{


    cudaError_t err;

    int total = class_length*values_length;
    int block_size = 1024;
    int grid_size = total/block_size+1;

    scatter_sum_kernel<<<grid_size, block_size, 0, stream>>>(index, values, sum_result, class_length, total);

    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

}