# NN-Grad
A toy implementation of neural network using numpy and cupy


## Benchmarks

<br>

### MNIST 10-Epochs
| Case       | Elapsed (s) | Train Acc. | Test Acc. |
| ---------- | ----------- | ---------- | --------- |
| numpy      | 29.577      | 99.613     | 98.110    |
| torch-cpu  | 18.076      | 99.597     | 98.100    |
| cupy       | 8.965       | 99.665     | 98.110    |
| torch-cuda | 5.426       | 99.482     | 97.890    |
