
import torch
from pynvrtc.compiler import Program
from cupy.cuda import function
from collections import namedtuple


kernel = '''
		extern "C" {
			__global__ void scatter_map(int* before_index, int* after_index, int* values, int length, int total)
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

			__global__ void scatter_sum(int* index, float* values, float* sum_result, int length, int total)
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
		}
		'''


def _initial_cupy(name="scatter_map"):
		program = Program(kernel.encode(), (name+'.cu').encode())
		ptx = program.compile()

		m = function.Module()
		m.load(bytes(ptx.encode()))

		kernel_function = m.get_function(name)

		Stream = namedtuple('Stream', ['ptr'])
		s = Stream(ptr=torch.cuda.current_stream().cuda_stream)
		return kernel_function, s



def scatter_map(before_index, after_index, values):

	class_len = len(before_index)
	index_len = len(values)
	total = index_len*class_len

	scatter_map, s = _initial_cupy(name="scatter_map")

	block_size = 1024 
	grid_size = total//block_size+1


	scatter_map(grid=(grid_size,1,1), block=(block_size,1,1), args=[before_index.data_ptr(), after_index.data_ptr(), values.contiguous().data_ptr(), class_len, total], stream=s)

	return values


def scatter_sum(index, values, index_len):
	class_len = index_len
	total = len(values)*class_len

	_scatter_sum, s = _initial_cupy(name="scatter_sum")

	block_size = 1024 
	grid_size = total//block_size+1

	index, values = index.int(), values.float()
	sum_result = torch.zeros(index_len).cuda(index.device).float()
	_scatter_sum(grid=(grid_size,1,1), block=(block_size,1,1), args=[index.data_ptr(), values.data_ptr(), sum_result.data_ptr(), class_len, total], stream=s)

	return sum_result