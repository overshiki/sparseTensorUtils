import copy
import torch
import numpy as np

r"""it seems pytorch now deprecated Variable and use tensor to handle backward method instead
however, in our current wrapping implementation, we just use autograd.Variable 
TODO: check tensor autograd method in future document of pytorch
"""


class TensorBase:
	r"""a wrapper to basic tensor type for all neural network framework like: numpy and cupy in chainer, tensor in pytorch
	"""
	def __init__(self):
		self.index_x = None 
		self.index_y = None
		self.value = None
		self.shape = None
		self.device = None
		self.empty = None

	def new(self, *args, **kwargs):
		r"""Constructs a new variable of the same data type as :attr:`self` variable.
		"""
		return self.__class__(*args, **kwargs)

	def to_dense(self):
		if self.empty==False:
			return self.torch().to_dense()
		else:
			return torch.zeros(self.shape).cuda(self.device)


	def numpy(self):
		data = self.to_dense()
		return data.numpy()

	# @property
	# def shape(self):
	# 	return self.shape

	@property
	def ndim(self):
		return len(self.shape)

	@property
	def dtype(self):
		return self.value.type()

	# @property
	# def size(self):
	# 	return self.ndarray.size()

	def __repr__(self):
		return "tensor with shape: {}".format(self.shape)
	def __str__(self):
		return self.__repr__()

	def torch(self):
		r'''
		transfer to pytorch sparse.FloatTensor
		'''
		if self.empty==False:
			i = torch.stack([self.index_x, self.index_y], dim=0).long()
			v = self.value.float()
			return torch.cuda.sparse.FloatTensor(i, v, torch.Size(self.shape)).cuda(self.device)
		else:
			return self.shape, "empty!"

	#TODO: currently FloatTensor is only support for pytorch, we wish in the future, more data type will be supported
	# def cast(self, dtype):
	# 	#transfer pytorch cast method into dtype-based method
	# 	if dtype=='float64':
	# 		self.value = self.value.double()
	# 	elif dtype=='float32':
	# 		self.value = self.value.float()
	# 	elif dtype=='float16':
	# 		self.value = self.value.half()
	# 	elif dtype=='int64':
	# 		self.value = self.value.long()
	# 	elif dtype=='int32':
	# 		self.value = self.value.int()
	# 	elif dtype=='int16':
	# 		self.value = self.value.short()
	# 	elif dtype=='int8':
	# 		self.value = self.value.char()
	# 	elif dtype=='bool':
	# 		self.value = self.value.byte()
	# 	else:
	# 		raise TypeError("unsupported dtype {}".format(dtype))


	# def astype(self, dtype):
	# 	if dtype=='float64':
	# 		return self.new(self.ndarray.double(), device=self.device)
	# 	elif dtype=='float32':
	# 		return self.new(self.ndarray.float(), device=self.device)
	# 	elif dtype=='float16':
	# 		return self.new(self.ndarray.half(), device=self.device)
	# 	elif dtype=='int64':
	# 		return self.new(self.ndarray.long(), device=self.device)
	# 	elif dtype=='int32':
	# 		return self.new(self.ndarray.int(), device=self.device)
	# 	elif dtype=='int16':
	# 		return self.new(self.ndarray.short(), device=self.device)
	# 	elif dtype=='int8':
	# 		return self.new(self.ndarray.char(), device=self.device)
	# 	elif dtype=='bool':
	# 		return self.new(self.ndarray.byte(), device=self.device)
	# 	else:
	# 		raise TypeError("unsupported dtype {}".format(dtype))

	# def reshape(self, *x):
	# 	return self.new(self.ndarray.reshape(*x), device=self.device)

	# # @device_guard_decorator
	# def sum(self, *x, axis=0):
	# 	return self.new(self.ndarray.sum(*x, dim=axis), device=self.device)

	# # @device_guard_decorator
	# def max(self, *x, **kwarg):
	# 	return self.new(self.ndarray.max(*x, **kwarg), device=self.device)

	# def transpose(self, *x):
	# 	return self.new(self.ndarray.transpose(*x), device=self.device)



	def __len__(self):
		return self.shape[0]

	# def __copy__(self):
	# 	result = self.new(self.ndarray.clone(), device=self.device)
	# 	return result

	# def __deepcopy__(self, memo):
	# 	result = self.new(self.ndarray.clone(), device=self.device)
	# 	memo[id(self)] = result
	# 	return result



class sparseTensor(TensorBase):
	def __init__(self, x, y, v, shape):
		super().__init__()
		self.index_x = x
		self.index_y = y
		self.value = v
		self.shape = shape
		self.device = self.index_x.device
		if len(x)>0:
			self.empty = False 
		else:
			self.empty = True