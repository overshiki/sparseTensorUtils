import torch
from kernels.c_kernel.scatter_utils import scatter_map as scatter_map_engine
def scatter_map_c(before_index, after_index, values):
	class_length = len(before_index)
	values_length = len(values)
	scatter_map_engine(before_index, after_index, values, class_length, values_length)
	return values

from kernels.c_kernel.scatter_utils import scatter_sum as scatter_sum_engine
def scatter_sum_c(index, values, index_len):
	class_length = index_len
	values_length = len(values)
	sum_result = torch.zeros(index_len).cuda(index.device).float()
	scatter_sum_engine(index, values, sum_result, class_length, values_length)
	return sum_result