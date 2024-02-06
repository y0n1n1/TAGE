# TAGE - Tensor-based Automatic Gradient Engine
A Tensor-based Automatic Gradient Engine in python using NumPy for fast computation -> Ready for Neural Network and Gradient Descent Training.
***
## Example
heres a simple example of using tensors and calling the .backward() function. TAGE generally features similar syntax as alternatives like PyTorch and Tinygrad 
```python
from tage import tensor

y = tensor([0,-5,-3,5,9,4,-2,1,-2,2,-9,-9,-7,6,5,7,-9,-9,7,0], requires=True)


x = tensor([-2,-1,-3,-2,3,-1,6,0,2,5,1,3,7,6,1,-6,-5,9,3,4],requires=True)

h = tensor([-2,-3,-4,-6,-4,-10,-3,5,9,8,-2,-8,7,-1,3,-9,-10,2,-5,2], requires=True)

b = tensor([1,2,4,9,4,25,2,62,0,16,1,16,12,0,2,20,25,1,6,1], requires=True)
z = ((x*y)/(-h))-b
j = z.sum()
j.backward()

print("TAGE")
print(f"{y.grad.numpy()}") # returns:
"""
[-1.         -0.33333334 -0.75       -0.33333334  0.75       -0.1
  2.         -0.         -0.22222222 -0.625       0.5         0.375
 -1.          6.         -0.33333334 -0.6666667  -0.5        -4.5
  0.6        -2.        ]
""" 
print(f"{x.grad.numpy()}") # returns:
"""
[ 0.         -1.6666667  -0.75        0.8333334   2.25        0.4
 -0.6666667  -0.2         0.22222222 -0.25       -4.5        -1.125
  1.          6.         -1.6666667   0.7777778  -0.90000004  4.5
  1.4        -0.        ]
""" 
print(f"{h.grad.numpy()}") # returns:
"""
[ -0.           0.5555556    0.5625      -0.2777778    1.6875
  -0.04        -1.3333334    0.          -0.04938272   0.15625
  -2.25        -0.421875    -1.          36.           0.5555556
  -0.5185185    0.45       -20.25         0.84         0.        ]
""" 
print(f"{b.grad.numpy()}") # returns:
"""
[-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.
 -1. -1.]
""" 
```

## Functions
Currently, TAGE contains the following functions:
* zeros
* ones
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
