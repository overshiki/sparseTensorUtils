from .math.sparse_op import slices, reduce_sum
from .tensor import TensorBase

def __getitem__(self, item):
	return slices(self, item)

def _sum(self, dim=-1):
	return reduce_sum(self, dim=dim)


def install_tensor_plugin():
	TensorBase.__getitem__ = __getitem__
	TensorBase.sum = _sum