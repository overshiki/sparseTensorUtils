#ifndef _SCATTER_CUDA_KERNEL
#define _SCATTER_CUDA_KERNEL

#ifdef __cplusplus
extern "C" {
#endif


void scatter_map_cuda(int* before_index, int* after_index, int* values, int class_length, int values_length, cudaStream_t stream);


void scatter_sum_cuda(int* index, float* values, float* sum_result, int class_length, int values_length, cudaStream_t stream);



#ifdef __cplusplus
}
#endif

#endif
