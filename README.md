# TAGE - Tensor-based Automatic Gradient Engine
A Tensor-based Automatic Gradient Engine in python using NumPy for fast computation -> Ready for Neural Network and Gradient Descent Training.
***
## Example
heres a simple example of using tensors and calling the .backward() function. TAGE generally features similar syntax as alternatives like PyTorch and Tinygrad 
```python
from tensor import tensor

# Perform Ops:
x = tensor.randn(5, 6)
y = tensor.rand(6, 3)
z = x.matmul(y)
w = tensor.rand(3)
j = z.dot(w)
k = (j - tensor.rand(5))*tensor.randint(-10, -3, (5))/(30*tensor.ones(5))
m = k/(-1)
l = m.sum()

# Call backward
l.backward()

# See grads
print(x.numpy())
print(f"GRAD: {x.grad.numpy()}")
print(y.numpy())
print(f"GRAD: {y.grad.numpy()}")
print(z.numpy())
print(f"GRAD: {z.grad.numpy()}")
print(j.numpy())
print(f"GRAD: {j.grad.numpy()}")
print(k.numpy())
print(f"GRAD: {k.grad.numpy()}")
print(m.numpy())
print(f"GRAD: {m.grad.numpy()}")
print(l.numpy())
print(f"GRAD: {l.grad.numpy()}")

```

## Functions
Currently, TAGE contains the following functions:
* zeros
* ones
* matmul
* randn
* eye
* transpose
* add
* sub
* mul
* pow
* truediv
* matmul
* tdot
* nroot
* sum
* neg
* sqrt
