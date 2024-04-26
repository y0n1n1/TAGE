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
print("TENSOR: ")
print(x.numpy())
# results
print("GRADIENT: ")
print(f"{x.grad.numpy()}")
print("\n \n")
print("TENSOR: ")
print(y.numpy())
print("GRADIENT: ")
print(f"{y.grad.numpy()}")
print("\n \n")
print("TENSOR: ")
print(z.numpy())
print("GRADIENT: ")
print(f" {z.grad.numpy()}")
print("\n \n")
print("TENSOR: ")
print(j.numpy())
print("GRADIENT: ")
print(f" {j.grad.numpy()}")
print("\n \n")
print("TENSOR: ")
print(k.numpy())
print("GRADIENT: ")
print(f"{k.grad.numpy()}")
print("\n \n")
print("TENSOR: ")
print(m.numpy())
print("GRADIENT: ")
print(f"{m.grad.numpy()}")
print("\n \n")
print("TENSOR: ")
print(l.numpy())
print("GRADIENT: ")
print(f"{l.grad.numpy()}")

```
And it outputs:
```python
TENSOR: 
[[-0.07525865 -0.8421788  -0.35153043 -0.9747891   0.4538262  -1.2204896 ]
 [-1.846694    0.4826549  -1.4315367   1.1357454   0.99839646  0.84751177]
 [-0.6494532  -0.1469772  -1.1433768   0.18571351  0.4385208   0.94998163]
 [ 0.6843067  -0.98688054 -0.14661244 -0.20684539  1.2123641   1.0443904 ]
 [ 0.7903265   0.6038584   0.82786953 -2.1770833  -0.16562416 -0.02135005]]
GRADIENT: 
[[0.21371137 0.23307114 0.36330381 0.07454167 0.13807976 0.16146255]
 [0.21371137 0.23307114 0.36330381 0.07454167 0.13807976 0.16146255]
 [0.21371137 0.23307114 0.36330381 0.07454167 0.13807976 0.16146255]
 [0.21371137 0.23307114 0.36330381 0.07454167 0.13807976 0.16146255]
 [0.21371137 0.23307114 0.36330381 0.07454167 0.13807976 0.16146255]]

 

TENSOR: 
[[0.8474021  0.3903513  0.24348201]
 [0.02789333 0.18242687 0.9315216 ]
 [0.7149148  0.89705586 0.8358888 ]
 [0.30531663 0.23369816 0.05183949]
 [0.46571922 0.9298308  0.02750619]
 [0.50554776 0.29351398 0.2744553 ]]
GRADIENT: 
[[-0.17107142 -0.0696241  -0.25566021]
 [-0.13874526 -0.05646772 -0.20734989]
 [-0.35019773 -0.14252644 -0.52335814]
 [-0.31776572 -0.12932698 -0.47488965]
 [ 0.45818013  0.18647403  0.68473403]
 [ 0.24957025  0.10157221  0.37297392]]

 

TENSOR: 
[[-1.0418594  -0.6624114  -1.4696918 ]
 [-1.3346642  -0.47446063 -0.8776961 ]
 [-0.6306746  -0.5760164  -0.9683615 ]
 [ 1.4769973   1.3410642  -0.5659726 ]
 [ 0.5257966   0.49226087  1.32367   ]]
GRADIENT: 
 [[0.1559771  0.06348088 0.23310227]
 [0.1559771  0.06348088 0.23310227]
 [0.1559771  0.06348088 0.23310227]
 [0.1559771  0.06348088 0.23310227]
 [0.1559771  0.06348088 0.23310227]]

 

TENSOR: 
[-2.3449078 -1.8980963 -1.5457032  0.7867722  1.807765 ]
GRADIENT: 
 [0.23333335 0.23333335 0.23333335 0.23333335 0.23333335]

 

TENSOR: 
[ 0.68936513  0.67059935  0.55800064  0.00458537 -0.3061334 ]
GRADIENT: 
[-1. -1. -1. -1. -1.]

 

TENSOR: 
[-0.68936513 -0.67059935 -0.55800064 -0.00458537  0.3061334 ]
GRADIENT: 
[1. 1. 1. 1. 1.]

 

TENSOR: 
-1.6164171000321708
GRADIENT: 
1.0
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

