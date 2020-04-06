# skipi
skipi is a library to easily define mathematical functions and apply various transforms on it. 

A function always consists of a domain and a map. Usually the domain is ommited since it's clear for the human what the domain is, however, not for the computer. 

This library aims to combine the domain and the map into one Function object and offer multiple convenient operations on it.

## Examples
### Algebraic operations
Supported features are: Addition, Subtraction, Multiplication, Division, Exponentiation, Composition
```python
import numpy as np
from skipi.function import Function

f = Function(np.linspace(0, 10, 100), lambda x: 2+x)
g = Function(f.get_domain(), lambda x: np.sin(x))
h1, h2, h3, h4, h5, h6 = f+g, f-g, f*g, g/f, f.composeWith(g), f**g
``` 

### Plotting
A function is plotted using matplotlib calling plot(). If you want to plot multiple functions into one graph, simply use
```python
g.plot()  # does not draw the graph yet
f.plot(show=True) # draws it
```

### Remeshing
If you want to re-mesh a function on a different domain/grid, you can use `remesh` or `vremesh`. 
The method `remesh` assigns a new mesh, independent of the previous one.
```python
f = Function(np.linspace(0, 10, 10), lambda x: np.sin(x))
f.remesh(np.linspace(0, 20, 1000))
```
However, if you want to restrict the domain, you can use `vremesh` which has a similar syntax as `slice` except that instead of indices we use values and it allows multiple slicing:
```python
f = Function(np.linspace(0, 10, 1000), lambda x: np.sin(x))
f.vremesh((np.pi, 2*np.pi)) # domain is now restricted to [pi, 2pi]
f.vremesh((None, 2*np.pi)) # domain is now restricted to [0, 2pi]
f.vremesh((np.pi, None)) # domain is now restricted to [pi, 10]
f.vremesh((0.5, 1.5), (2.0, 2.5)) # domain is now restricted to [0.5, 1.5] union [2.0, 2.5]
```
### Creating functions from data
If you don't have an analytical formulation of `y = f(x)`, but rather have y_i and x_i values, then you can create a function by interpolation. By default, linear interpolation is used.
```python
x_i = np.linspace(0, 10, 100)
y_i = np.sin(x_i)

f = Function.to_function(x_i, y_i)
print(f(0.1234)) # linearly interpolated, not sin(0.1234)!
```

### Integration
Calculate the integral function of `f(x) = 5x`

```python
import numpy as np
from skipi.function import Function, Integral

f = Function(np.linspace(0, 10, 100), lambda x: 5*x)
F = Integral.from_function(f) # Integral function
F.plot(show=True)
```

### Fourier transform
Calculate the fourier transform (analytical fourier transform, not fft) of f(x) = exp(-x^2)

```python
from skipi.fourier import FourierTransform, InverseFourierTransform

t_space, freq_space = np.linspace(-5, 5, 100), np.linspace(-10, 10, 100)
f = Function(t_space, lambda x: np.exp(-x**2))
F = FourierTransform.from_function(freq_space, f)
f2 = InverseFourierTransform.from_function(t_space, F)

# f2 should be equal to f
(f-f2).plot(show=True)
```
