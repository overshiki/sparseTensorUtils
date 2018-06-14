## sparse tensor utils based on pytorch tensor and sparse.Tensor. Currently does not guaranteed to support graph backward, so that should only be used for gpu boosted inference.

### description:

Here provides a basic sparse tensor wrapper: 
```python
In [1]: from sparseTensorUtils.sparse.tensor import sparseTensor
```
The basic functions for the wrapper is written with pytorch built-in function and some c cuda kernel, linked with pytorch cffi, so that fully support gpus.

### how to install:

To build the c kernel:
```bash
cd kernels/c_kernel
make
```

### dependencies:
        torch

The object can be constructed with 1d tensor indices_x, 1d tensor indices_y, 1d tensor values and shape tuple:
```python
In [2]: import torch
In [3]: data = torch.randn(100,100).cuda(0)
In [4]: indices = torch.nonzero(data>0)
In [5]: value = torch.masked_select(data, data>0)
In [6]: x, y, v, shape = indices[:, 0], indices[:, 1], value, list(data.shape)
In [7]: sp = sparseTensor(x, y, v, shape)
```
Most importantly, the sparseTensor wrapper supports slice and reduce_sum:
```python
sp_test = sp[5:50:10, 5:50:10]

In [18]: sp_test.sum()
Out[18]: tensor(8.5090, device='cuda:0')

In [19]: sp_test.sum(dim=0)
Out[19]: tensor([ 1.0469,  3.2423,  2.3594,  0.3194,  1.5410], device='cuda:0')

In [20]: sp_test.sum(dim=1)
Out[20]: tensor([ 0.3688,  1.0673,  2.0283,  1.3304,  3.7141], device='cuda:0')
```

to transform the wrapper object into pytorch sparseTensor, do:
```python
In [10]: sp_test.torch()
Out[10]: 
torch.cuda.sparse.FloatTensor of size (5,5) with indices:
tensor([[ 0,  1,  1,  2,  2,  3,  3,  4,  4,  4],
        [ 1,  0,  2,  2,  4,  1,  3,  1,  2,  4]], device='cuda:0')
and values:
tensor([ 0.3688,  1.0469,  0.0204,  0.6837,  1.3446,  1.0110,  0.3194,
         1.8625,  1.6553,  0.1964], device='cuda:0')
```
and to transform it into pytorch dense tensor, do:
```python
In [11]: sp_test.to_dense()
Out[11]: 
tensor([[ 0.0000,  0.3688,  0.0000,  0.0000,  0.0000],
        [ 1.0469,  0.0000,  0.0204,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.6837,  0.0000,  1.3446],
        [ 0.0000,  1.0110,  0.0000,  0.3194,  0.0000],
        [ 0.0000,  1.8625,  1.6553,  0.0000,  0.1964]], device='cuda:0')
```



and all kinds of element-wise math operation, to list a few:
```python 
In [13]: (sp_test+10).to_dense()
Out[13]: 
tensor([[  0.0000,  10.3688,   0.0000,   0.0000,   0.0000],
        [ 11.0469,   0.0000,  10.0204,   0.0000,   0.0000],
        [  0.0000,   0.0000,  10.6837,   0.0000,  11.3446],
        [  0.0000,  11.0110,   0.0000,  10.3194,   0.0000],
        [  0.0000,  11.8625,  11.6553,   0.0000,  10.1964]], device='cuda:0')

In [14]: (sp_test*10).to_dense()
Out[14]: 
tensor([[  0.0000,   3.6880,   0.0000,   0.0000,   0.0000],
        [ 10.4688,   0.0000,   0.2042,   0.0000,   0.0000],
        [  0.0000,   0.0000,   6.8371,   0.0000,  13.4464],
        [  0.0000,  10.1100,   0.0000,   3.1944,   0.0000],
        [  0.0000,  18.6250,  16.5527,   0.0000,   1.9638]], device='cuda:0')

In [15]: (sp_test-sp_test).to_dense()
Out[15]: 
tensor([[ 0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.]], device='cuda:0')

In [16]: (sp_test/10).to_dense()
Out[16]: 
tensor([[ 0.0000,  0.0369,  0.0000,  0.0000,  0.0000],
        [ 0.1047,  0.0000,  0.0020,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0684,  0.0000,  0.1345],
        [ 0.0000,  0.1011,  0.0000,  0.0319,  0.0000],
        [ 0.0000,  0.1862,  0.1655,  0.0000,  0.0196]], device='cuda:0')

In [17]: (sp_test**10).to_dense()
Out[17]: 
tensor([[   0.0000,    0.0000,    0.0000,    0.0000,    0.0000],
        [   1.5811,    0.0000,    0.0000,    0.0000,    0.0000],
        [   0.0000,    0.0000,    0.0223,    0.0000,   19.3226],
        [   0.0000,    1.1156,    0.0000,    0.0000,    0.0000],
        [   0.0000,  502.2888,  154.4099,    0.0000,    0.0000]], device='cuda:0')
```

#### future work
The future work include element-wise comparision and some broadcasting rules
