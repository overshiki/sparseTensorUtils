import torch
from .. import np, sparseTensor

from pynvrtc.compiler import Program
from cupy.cuda import function
from collections import namedtuple

from .. import scatter_map_c, scatter_sum_c
import math

kernel = '''
		extern "C"
		__global__ void indices_slice(int* indices, int start, int stop, int step, int* binary, int total)
		{
			int i = blockIdx.x * blockDim.x + threadIdx.x;
			
			if(i < total){
				int index = indices[i];

				if((index>=start)&&(index<stop)&&((index-start)/step)*step==(index-start)){
					binary[i] = 1;
				}else{
					binary[i] = 0;
				}
				
				
			}

		}
		'''

def _initial_cupy(name="indices_slice"):
		program = Program(kernel.encode(), (name+'.cu').encode())
		ptx = program.compile()

		m = function.Module()
		m.load(bytes(ptx.encode()))

		kernel_function = m.get_function(name)

		Stream = namedtuple('Stream', ['ptr'])
		s = Stream(ptr=torch.cuda.current_stream().cuda_stream)
		return kernel_function, s


def torch2tensorUtils(data):
	shape = list(data.shape)
	i, v = data._indices(), data._values()
	index_x, index_y = list(map(lambda x:x.squeeze(), i.chunk(2)))
	return sparseTensor(index_x, index_y, v, shape)

def sparse_add(sparse_A, sparse_B):
	result = sparse_A.torch()+sparse_B.torch()
	return torch2tensorUtils(result)

def sparse_add_scalar(sparse_A, scalar):
	return sparseTensor(sparse_A.index_x, sparse_A.index_y, sparse_A.value+scalar, sparse_A.shape)


def sparse_sub(sparse_A, sparse_B):
	result = sparse_A.torch()-sparse_B.torch()
	return torch2tensorUtils(result)

def sparse_sub_scalar(sparse_A, scalar):
	return sparseTensor(sparse_A.index_x, sparse_A.index_y, sparse_A.value-scalar, sparse_A.shape)

def sparse_neg(sparse_A):
	return sparseTensor(sparse_A.index_x, sparse_A.index_y, sparse_A.value*(-1), sparse_A.shape)



def sparse_mul(sparse_A, sparse_B):
	result = sparse_A.torch()*sparse_B.torch()
	return torch2tensorUtils(result)

def sparse_mul_scalar(sparse_A, scalar):
	return sparseTensor(sparse_A.index_x, sparse_A.index_y, sparse_A.value*scalar, sparse_A.shape)



def sparse_div_scalar(sparse_A, scalar):
	return sparseTensor(sparse_A.index_x, sparse_A.index_y, sparse_A.value/scalar, sparse_A.shape)



def sparse_pow_scalar(sparse_A, scalar):
	return sparseTensor(sparse_A.index_x, sparse_A.index_y, sparse_A.value**scalar, sparse_A.shape)

def rsparse_pow_scalar(sparse_A, scalar):
	return sparseTensor(sparse_A.index_x, sparse_A.index_y, scalar**sparse_A.value, sparse_A.shape)



def sparse_abs(sparse_A):
	return sparseTensor(sparse_A.index_x, sparse_A.index_y, torch.abs(sparse_A.value), sparse_A.shape)


def sparse_eq(sparse_A, sparse_B):
	raise ValueError("sparse_eq TODO")

def sparse_lt(sparse_A, sparse_B):
	raise ValueError("sparse_lt TODO")

def sparse_le(sparse_A, sparse_B):
	raise ValueError("sparse_le TODO")

def sparse_gt(sparse_A, sparse_B):
	raise ValueError("sparse_gt TODO")

def sparse_ge(sparse_A, sparse_B):
	raise ValueError("sparse_ge TODO")

def sparse_ne(sparse_A, sparse_B):
	raise ValueError("sparse_ne TODO")






















def get_slice(sparse_A, slice_A, dim=0):
	_len = len(sparse_A)
	start, stop, step = 0, _len, 1
	if slice_A.start is not None:
		start = int(slice_A.start)
	if slice_A.stop is not None:
		stop = int(slice_A.stop)
	if slice_A.step is not None:
		step = int(slice_A.step)

	if dim==0:
		sparse_index = sparse_A.index_x.int().contiguous()
	elif dim==1:
		sparse_index = sparse_A.index_y.int().contiguous()


	total = len(sparse_index)

	indices_slice, s = _initial_cupy(name="indices_slice")

	block_size = 1024 
	grid_size = total//block_size+1

	binary = torch.zeros_like(sparse_A.index_x).int()


	indices_slice(grid=(grid_size,1,1), block=(block_size,1,1), args=[sparse_index.data_ptr(), start, stop, step, binary.data_ptr(), total], stream=s)


	binary = binary.byte()

	sparse_index_x = torch.masked_select(sparse_A.index_x, binary)
	sparse_index_y = torch.masked_select(sparse_A.index_y, binary)
	sparse_value = torch.masked_select(sparse_A.value, binary)

	############################################################################
	#now remap indices
	############################################################################
	indices = torch.IntTensor([x for x in range(start, stop, step)]).cuda(sparse_A.device).int()
	new_indices = torch.IntTensor([x for x in range(0, math.ceil((stop-start)/step), 1)]).cuda(sparse_A.device).int()


	shape = sparse_A.shape 
	if dim==0:
		sparse_input = sparse_index_x.int()
		sparse_index_x = scatter_map(indices, new_indices, sparse_input).long()
		shape[0] = len(new_indices)
	elif dim==1:
		sparse_input = sparse_index_y.int()
		sparse_index_y = scatter_map(indices, new_indices, sparse_input).long()
		shape[1] = len(new_indices)


	return sparseTensor(sparse_index_x, sparse_index_y, sparse_value, shape)

def slices(sparse_A, slice_A):
	if isinstance(slice_A, slice):
		return get_slice(sparse_A, slice_A, dim=0)

	elif isinstance(slice_A, tuple):
		slice_x, slice_y = slice_A
		sparse_A = get_slice(sparse_A, slice_x, dim=0)
		sparse_A = get_slice(sparse_A, slice_y, dim=1)
		return sparse_A


def reduce_sum(sparse_A, dim=-1):
	if dim==-1:
		return sparse_A.value.sum()
	elif dim==1:
		index_len = sparse_A.shape[0]
		index = sparse_A.index_x.int()
		values = sparse_A.value.float()
		return scatter_sum_c(index, values, index_len)
	elif dim==0:
		index_len = sparse_A.shape[1]
		index = sparse_A.index_y.int()
		values = sparse_A.value.float()
		return scatter_sum_c(index, values, index_len)		

