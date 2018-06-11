from sparseTensorUtils.sparse.tensor import sparseTensor
import torch

data = torch.randn(100,100).cuda(0)
# print(data.shape)

indices = torch.nonzero(data>0)
# print(indices.shape)
value = torch.masked_select(data, data>0)
# print(value.shape)

x, y, v, shape = indices[:, 0], indices[:, 1], value, list(data.shape)

sp = sparseTensor(x, y, v, shape)


# print(sp.torch())
# print(sp.to_dense())
# print((sp+sp).to_dense())
# print((sp+5).to_dense())

# print(sp[5:50:5])
# print(sp[5:50:5].index_x)

# print(sp[5:50:5].to_dense())

sp_test = sp[5:50:10, 5:50:10]
print((sp_test*-1).to_dense())


print(abs(sp_test).to_dense())

