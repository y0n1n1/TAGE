# TAGE - Tensor-based Automatic Gradient Engine
A Tensor-based Automatic Gradient Engine in python using NumPy for fast computation -> Ready for Neural Network and Gradient Descent Training.
***
## Example
heres a simple example of using tensors and calling the .backward() function. TAGE generally features similar syntax as alternatives like PyTorch and Tinygrad 
```python
from tensor import Tensor

# calculate output
x = Tensor.randn(5, 6)
y = Tensor.rand(6, 3)
z = x.matmul(y)
w = Tensor.rand(3)
j = z.dot(w)
k = (j - Tensor.rand(5))*Tensor.randint(-10, -3, (5))/(30*Tensor.ones(5))
m = k/(-1)
l = m.sum()
# call backward() with respect to the output
l.backward()

# results
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
* randn
* rand
* randint
* eye
* normal
* uniform
* numpy
* transpose
* add
* sub
* mul
* pow
* truediv
* matmul
* dot
* tdot
* nroot
* sum
* neg
* sqrt

