


int scatter_sum(THCudaIntTensor *index, THCudaTensor *values, THCudaTensor *sum_result, int class_length, int values_length);

int scatter_map(THCudaIntTensor *before_index, THCudaIntTensor *after_index, THCudaTensor *values, int class_length, int values_length);


