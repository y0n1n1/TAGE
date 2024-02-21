import numpy as np 
from typing import Optional

DEPENDENCIES_CTX_DEBUG = True

class Tensor:
    ########## SETUP Tensor ########## 
    def __init__(self, data, dtype=np.float32, requires:Optional[bool]=True, dependencies=[]):
        if isinstance(data,list) or isinstance(data, int) or isinstance(data, float) or isinstance(data, np.float32): self.data=np.array(data, dtype=dtype) 
        elif isinstance(data,np.ndarray): self.data=data 
        else: raise Exception(f'Tensor cannot be created from {data}')
        self.dtype = dtype
        self.grad, self.requires = None, requires
        if requires: self.dependencies = dependencies
        if requires: self.ctx = lambda: None
        else: self.ctx = None
        self.shape = np.shape(self.data)
    def __repr__(self):return f"<Tensor {self.data}, requires?={self.requires}, Grad: {self.grad}>" if (not DEPENDENCIES_CTX_DEBUG) else f"<Tensor {self.data}, grad={self.requires}, dependencies={[i.data for i in self.dependencies]}, ctx=None >"
    def __str__(self):return f"<Tensor {self.data}, requires?={self.requires}, Grad: {self.grad}>" if (not DEPENDENCIES_CTX_DEBUG) else f"<Tensor {self.data}, grad={self.requires}, dependencies={[i.data for i in self.dependencies]}>"
    def detach(self): return Tensor(self.data)

    ########## STATICMETHODS ########## 
    @staticmethod
    def populate(function, shape, requires=True):
        # function must have no inputs. For example: a random number generator
        base = np.ones(shape)
        def create(x):
            return x*function()
        return Tensor(create(base), requires=requires)
    @staticmethod
    def zeros(*args, requires=True): return Tensor(np.zeros(*args, dtype=np.float32), requires=requires)
    @staticmethod
    def ones(*args, requires=True): return Tensor(np.ones(*args, dtype=np.float32), requires=requires)
    @staticmethod
    def randn(*args, requires=True): return Tensor(np.random.randn(*args).astype(np.float32), requires=requires)
    @staticmethod
    def rand(*args, requires=True): return Tensor(np.random.rand(*args).astype(np.float32), requires=requires)
    @staticmethod
    def randint(low, high, shape, requires=True): 
        def intf():
            return np.random.randint(low, high)
        return Tensor.populate(intf, shape, requires=requires)
    @staticmethod
    def eye(dim, requires=True): return Tensor(np.eye(dim).astype(np.float32), requires=requires)
    @staticmethod
    def normal(shape, mean=0.0, std=1.0, requires=True): return (std * Tensor.randn(shape, requires=requires)) + mean
    @staticmethod
    def uniform(shape, low=0.0, high=1.0, requires=True):
        return ((high-low) * Tensor.rand(shape, requires=requires)) + low

    ########## UTILS ########## 
    def numpy(self): return self.data
    def __getitem__(self, indices):
        if self.grad == None:
            return Tensor(self.data.__getitem__(indices))
        else: 
            x = Tensor(self.data.__getitem__(indices))
            x.grad = Tensor(self.grad.data.__getitem__(indices))
            return x
    def transpose(self): 
        out = Tensor(self.data.transpose(), dependencies=self.dependencies)
        out.ctx = self.ctx
        return out
    def T(self): self.transpose()

    

    ########## AUTOMATIC DIFFERENTIATION ########## 
    def topo(self):
            def build_topo(node:Tensor, visited:set):
                if node not in visited and (node.ctx != None):
                    yield node
                    visited.add(node)
                    for child in node.dependencies: yield from build_topo(child, visited)
            return list(build_topo(self, set())) 
    def backward(self):
        assert self.shape == tuple(), f"backward can only be called for scalar Tensors, but it has shape {self.shape})"
        self.grad = Tensor(1.0, requires=False)
        for t0 in self.topo():
            if t0.ctx is None: raise RuntimeError("Tensor doesn't require grad yet its in the backward pass")
            t0.ctx()
            # reset ctx
            t0.ctx = lambda: None
        return self

    ########## OPS ##########

    ########## 2 ITEM OPS ##########
    def add(self, x): 
        if isinstance(x, float) or isinstance(x, int):out = Tensor(self.data.__add__(x), dependencies=[self])
        else:out = Tensor(self.data.__add__(x.data), dependencies=[self, x])
        def backward():
            self.grad =  out.grad
            if (not (isinstance(x, float) or isinstance(x, int))): x.grad = out.grad
        self.ctx = backward
        if (not (isinstance(x, float) or isinstance(x, int))): x.ctx = backward
        return out
    def sub(self, x): 
        if isinstance(x, float) or isinstance(x, int):
            out = Tensor(self.data.__sub__(x), dependencies=[self])
        else:
            out = Tensor(self.data.__sub__(x.data), dependencies=[self, x])
        def backward():
            self.grad = out.grad
            if (not (isinstance(x, float) or isinstance(x, int))):x.grad = out.grad.neg()
        self.ctx = backward
        if (not (isinstance(x, float) or isinstance(x, int))):x.ctx = backward
        return out
    def mul(self, x): 
        if isinstance(x, Tensor):
            out = Tensor(self.data.__mul__(x.data), dependencies=[self, x])
            def backward():
                self.grad = Tensor(out.grad.data.__mul__(x.data))
                x.grad = Tensor(out.grad.data.__mul__(self.data))
            self.ctx = backward
            x.ctx = backward
            return out
        else:
            out = Tensor(self.data.__mul__(x), dependencies=[self])
            def backward():
                self.grad = Tensor(out.grad.data.__mul__(x))
            self.ctx = backward
            return out
    def pow(self, x): 
        return Tensor(self.data.__pow__(x), dependencies=[self])
    def truediv(self, x): 
        if (not (isinstance(x, float) or isinstance(x, int))):out = Tensor(self.data.__truediv__(x.data), dependencies=[self, x])
        else: out = Tensor(self.data.__truediv__(x), dependencies=[self])
        def backward():
            self.grad = Tensor(1/x.data)*out.grad if (not (isinstance(x, float) or isinstance(x, int))) else Tensor(1/x)*out.grad
            if (not (isinstance(x, float) or isinstance(x, int))):x.grad = (-Tensor(self.data.__truediv__(np.power(x.data, 2))))*out.grad
        self.ctx = backward
        if (not (isinstance(x, float) or isinstance(x, int))):x.ctx = backward
        return out
    def matmul(self, x): 
        out = Tensor(self.data.__matmul__(x.data), dependencies=[self, x])
        def backward():
            self.grad = out.grad @ x.transpose()
            x.grad = self.transpose()@ out.grad 
        self.ctx = backward
        x.ctx = backward
        return out
    def dot(self, x):
        SS = len(self.shape)
        XX = len(x.shape)
        if ((SS==2) and (XX==2)): 
            return self.matmul(x)
        elif (SS==2) and (XX==1):
            out = Tensor(self.data.dot(x.data), dependencies=[self, x])
            def backward():
                self.grad = Tensor(np.expand_dims(out.grad.data, axis=1))@Tensor(np.expand_dims(x.data, axis=0))
                x.grad = self.transpose()@out.grad
            self.ctx = backward
            x.ctx = backward
            return out
        elif (SS==1) and (XX==2):
            out = Tensor(x.data.dot(self.data), dependencies=[self, x])
            def backward():
                self.grad = x.transpose()@out.grad 
                x.grad = Tensor(np.expand_dims(out.grad.data, axis=1))@Tensor(np.expand_dims(self.data, axis=0))
            self.ctx = backward
            x.ctx = backward
            return out
        else:
            out = Tensor(self.data.dot(x.data), dependencies=[self, x])
            print(f"GRAD NOT SUPPORTED FOR DIMS: {SS}, {XX}")
    def tdot(self, x): return Tensor(np.Tensordot(self.data, x.data, axes=0), dependencies=[self, x])
    def nroot(self, n):
        out = Tensor(np.power(self.data, (1/n)), dependencies=[self])
        def backward():
            locg = 1/(n*(np.power(out.data, n-1)))
            self.grad = Tensor(out.grad.data.__mul__(locg)) 
        self.ctx = backward
        return out

    ########## 1 ITEM OPS ##########
    def sum(self):
        out = Tensor(np.array(self.data.sum()), dependencies=[self])
        def backward():
            self.grad = Tensor.ones(self.shape) * out.grad
        self.ctx = backward
        return out
    def neg(self):
        out = Tensor(self.data.__neg__(), dependencies=[self])
        def backward():
            self.grad = out.grad.neg()
        self.ctx = backward
        return out
    def sqrt(self): return self.nroot(2)

    ########## __OPS__ ##########
    def __add__(self, x): return self.add(x)
    def __sub__(self, x): return self.sub(x)
    def __mul__(self, x): return self.mul(x)
    def __pow__(self, x): return self.pow(x)
    def __truediv__(self, x): return self.truediv(x)
    def __matmul__(self, x): return self.matmul(x)
    def __neg__(self): return self.neg()
    def __radd__(self, x): return self.add(x)
    def __rsub__(self, x): return self.sub(x)
    def __rmul__(self, x): return self.mul(x)
    def __rpow__(self, x): return self.pow(x)
    def __rtruediv__(self, x): return self.truediv(x)
    def __rmatmul__(self, x): return self.matmul(x)

