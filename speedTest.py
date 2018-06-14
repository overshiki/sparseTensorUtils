from sparseTensorUtils.sparse.tensor import sparseTensor
import torch
from timeit import default_timer


for size in range(100, 100000, 500):
	data = torch.randn(size, size).cuda(0)
	indices = torch.nonzero(data>0)
	value = torch.masked_select(data, data>0)


	x, y, v, shape = indices[:, 0], indices[:, 1], value, list(data.shape)
	sp = sparseTensor(x, y, v, shape)

	sp_mean, dense_mean = 0, 0
	for i in range(1000):
		end = default_timer()
		_sum = sp.sum(dim=0)
		rtime = default_timer()-end
		# print("sparse: ", rtime, end="\t")
		sp_mean += rtime

		end = default_timer()
		_sum = data.sum(dim=0)
		rtime = default_timer()-end
		# print("dense: ", rtime)
		dense_mean += rtime

	sp_mean, dense_mean = sp_mean/1000, dense_mean/1000 
	print("[size]: ", size, "sparse: ", sp_mean, "dense: ", dense_mean)