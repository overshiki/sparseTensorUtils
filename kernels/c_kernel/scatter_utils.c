#include <THC/THC.h>
#include "scatter_cuda_kernel.h"

extern THCState *state;


int scatter_sum(THCudaIntTensor *index, THCudaTensor *values, THCudaTensor *sum_result, int class_length, int values_length)
{
    int *k_index = THCudaIntTensor_data(state, index);
    float *k_values = THCudaTensor_data(state, values);
    float *k_sum_result = THCudaTensor_data(state, sum_result);
    cudaStream_t stream = THCState_getCurrentStream(state);

    scatter_sum_cuda(k_index, k_values, k_sum_result, class_length, values_length, stream);

    return 1;
}



int scatter_map(THCudaIntTensor *before_index, THCudaIntTensor *after_index, THCudaIntTensor *values, int class_length, int values_length)
{
    int *k_before_index = THCudaIntTensor_data(state, before_index);
    int *k_after_index = THCudaIntTensor_data(state, after_index);
    int *k_values = THCudaIntTensor_data(state, values);
    
    cudaStream_t stream = THCState_getCurrentStream(state);

    scatter_map_cuda(k_before_index, k_after_index, k_values, class_length, values_length, stream);

    return 1;
}