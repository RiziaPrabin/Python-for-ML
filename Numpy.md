# NumPy: A Technical Overview

## Introduction

NumPy, short for Numerical Python, is a fundamental library in Python for numerical computing, widely used in scientific computing, data analysis, and machine learning applications. Its primary data structure, `numpy.ndarray`, is an n-dimensional array that efficiently stores and manipulates numerical data. In this technical overview, we'll delve into the key features and capabilities of NumPy that make it such a powerful tool for numerical computation.

## Key Features

### Arrays and Memory Management

NumPy's `ndarray` is the cornerstone of its functionality. Unlike Python lists, `ndarray` is a contiguous block of memory containing elements of the same data type, allowing for efficient storage and manipulation of large datasets. This layout facilitates fast element-wise operations and eliminates the overhead associated with dynamic typing.

### Vectorized Operations and Performance

NumPy's vectorized operations enable efficient element-wise operations on arrays, leveraging optimized, low-level implementations for improved performance. By avoiding explicit iteration over array elements, vectorized operations take advantage of hardware-level optimizations and parallelism, resulting in significant speedups compared to traditional Python loops.

### Broadcasting

Broadcasting is a powerful mechanism in NumPy that allows arrays with different shapes to be combined in arithmetic operations. NumPy automatically aligns dimensions to perform operations, simplifying code and improving readability. This feature is particularly useful when working with arrays of different shapes or when performing operations between arrays and scalars.

### Mathematical Functions and Numerical Computations

NumPy provides a comprehensive suite of mathematical functions for numerical computations. These functions cover a wide range of mathematical operations, including basic arithmetic, trigonometry, logarithms, exponentials, and more. NumPy's ability to apply these functions element-wise to entire arrays enables efficient numerical computations on large datasets.

### Random Number Generation

The `numpy.random` module offers robust tools for generating random numbers from various probability distributions. Random number generation is essential for tasks such as model initialization, data simulation, and statistical analysis. NumPy's random number generation capabilities provide researchers and practitioners with tools to simulate stochastic processes and analyze probabilistic outcomes effectively.

### Integration with Low-Level Languages and Existing Libraries

NumPy seamlessly integrates with code written in low-level languages such as C and Fortran, allowing Python developers to leverage existing optimized numerical routines. This integration enhances computational efficiency and enables access to a vast ecosystem of pre-existing algorithms and numerical methods. NumPy's interoperability with other libraries, such as SciPy, Matplotlib, and scikit-learn, further extends its capabilities for scientific computing and machine learning applications.


## Importing NumPy
Once you've installed NumPy you can import it as a library:
```python
import numpy as np
import numpy as np
```
## NumPy Arrays
 NumPy arrays essentially come in two flavors: vectors and matrices. Vectors are strictly 1-dimensional (1D) arrays and matrices are 2D (but you should note a matrix can still have only one row or one column).

### Why use Numpy array? Why not just a list?
There are lot's of reasons to use a Numpy array instead of a "standard" python list object. 
Our main reasons are:

> Memory Efficiency of Numpy Array vs list
> Easily expands to N-dimensional objects
> Speed of calculations of numpy array
> Broadcasting operations and functions with numpy
> All the data science and machine learning libraries we use are built with Numpy

### Simple Example of what numpy array can do
```python
my_list = [1,2,3]
my_array = np.array([1,2,3])
type(my_list)
```
Output list
​
Let's begin our introduction by exploring how to create NumPy arrays.

### Creating NumPy Arrays from Objects
From a Python List
We can create an array by directly converting a list or list of lists:
```python
my_list = [1,2,3]
my_list
```
Output [1, 2, 3]
```python
np.array(my_list)
```
Output array([1, 2, 3])
```python
my_matrix = [[1,2,3],[4,5,6],[7,8,9]]
my_matrix
```
Output [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
np.array(my_matrix)
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
### Built-in Methods to create arrays
There are lots of built-in ways to generate arrays.

#### arange
Return evenly spaced values within a given interval. [reference]
```python
np.arange(0,10)
```
Output array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```python
np.arange(0,11,2)
```
Output array([ 0,  2,  4,  6,  8, 10])
#### zeros and ones
Generate arrays of zeros or ones. [reference]
```python
np.zeros(3)
```
Output array([0., 0., 0.])
```python
np.zeros((5,5))
```
output
array([[0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]])
```python
np.ones(3)
```
output array([1., 1., 1.])
```python
np.ones((3,3))
```
output 
array([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]])
#### linspace
Return evenly spaced numbers over a specified interval. [reference]
```python
np.linspace(0,10,3)
```
Output array([ 0.,  5., 10.])
``` python
np.linspace(0,5,20)
```
Output
array([0.        , 0.26315789, 0.52631579, 0.78947368, 1.05263158,
       1.31578947, 1.57894737, 1.84210526, 2.10526316, 2.36842105,
       2.63157895, 2.89473684, 3.15789474, 3.42105263, 3.68421053,
       3.94736842, 4.21052632, 4.47368421, 4.73684211, 5.        ])
Note that .linspace() includes the stop value. To obtain an array of common fractions, increase the number of items:
```python
np.linspace(0,5,21)
```
Output 
array([0.  , 0.25, 0.5 , 0.75, 1.  , 1.25, 1.5 , 1.75, 2.  , 2.25, 2.5 ,
       2.75, 3.  , 3.25, 3.5 , 3.75, 4.  , 4.25, 4.5 , 4.75, 5.  ])
#### eye
Creates an identity matrix [reference]
```python
np.eye(4)
```
Output
array([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]])
#### Random
Numpy also has lots of ways to create random number arrays:

##### rand
Creates an array of the given shape and populates it with random samples from a uniform distribution over [0, 1). [reference]
```python
np.random.rand(2)
```
Output array([0.37065108, 0.89813878])
``` python
np.random.rand(5,5)
```
Output
array([[0.03932992, 0.80719137, 0.50145497, 0.68816102, 0.1216304 ],
       [0.44966851, 0.92572848, 0.70802042, 0.10461719, 0.53768331],
       [0.12201904, 0.5940684 , 0.89979774, 0.3424078 , 0.77421593],
       [0.53191409, 0.0112285 , 0.3989947 , 0.8946967 , 0.2497392 ],
       [0.5814085 , 0.37563686, 0.15266028, 0.42948309, 0.26434141]])
##### randn
Returns a sample (or samples) from the "standard normal" distribution [σ = 1]. Unlike rand which is uniform, values closer to zero are more likely to appear. [reference]
```python
np.random.randn(2)
```
Output array([-0.36633217, -1.40298731])
``` python
np.random.randn(5,5)
```
Output
array([[-0.45241033,  1.07491082,  1.95698188,  0.40660223, -1.50445807],
       [ 0.31434506, -2.16912609, -0.51237235,  0.78663583, -0.61824678],
       [-0.17569928, -2.39139828,  0.30905559,  0.1616695 ,  0.33783857],
       [-0.2206597 , -0.05768918,  0.74882883, -1.01241629, -1.81729966],
       [-0.74891671,  0.88934796,  1.32275912, -0.71605188,  0.0450718 ]])
##### randint
Returns random integers from low (inclusive) to high (exclusive). [reference]
```python
np.random.randint(1,100)
```
Output 61
``` python
np.random.randint(1,100,10)
```
Output array([39, 50, 72, 18, 27, 59, 15, 97, 11, 14])
##### seed
Can be used to set the random state, so that the same "random" results can be reproduced. [reference]
``` python
np.random.seed(42)
np.random.rand(4)
```
Output array([0.37454012, 0.95071431, 0.73199394, 0.59865848])
``` python
np.random.seed(42)
np.random.rand(4)
```
Output array([0.37454012, 0.95071431, 0.73199394, 0.59865848])

### Array Attributes and Methods
Let's discuss some useful attributes and methods for an array:
```python
arr = np.arange(25)
ranarr = np.random.randint(0,50,10)
arr
```
Output
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24])
ranarr
array([38, 18, 22, 10, 10, 23, 35, 39, 23,  2])

#### Reshape
Returns an array containing the same data with a new shape. [reference]
``` python
arr.reshape(5,5)
```
Output 
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24]])
### max, min, argmax, argmin
These are useful methods for finding max or min values. Or to find their index locations using argmin or argmax

#### ranarr
``` python
array([38, 18, 22, 10, 10, 23, 35, 39, 23,  2])
ranarr.max()
```
39
``` python
ranarr.argmax()
```
7
``` python
ranarr.min()
```
2
``` python
ranarr.argmin()
```
9
#### Shape
Shape is an attribute that arrays have (not a method): [reference]
``` python
# Vector
arr.shape
```
(25,)
Notice the two sets of brackets
``` python
arr.reshape(1,25)
```
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24]])
 ```python
arr.reshape(1,25).shape
```
(1, 25)
``` python
arr.reshape(25,1)
```
array([[ 0],
       [ 1],
       [ 2],
       [ 3],
       [ 4],
       [ 5],
       [ 6],
       [ 7],
       [ 8],
       [ 9],
       [10],
       [11],
       [12],
       [13],
       [14],
       [15],
       [16],
       [17],
       [18],
       [19],
       [20],
       [21],
       [22],
       [23],
       [24]])
 ```python
arr.reshape(25,1).shape
```
(25, 1)
#### dtype
You can also grab the data type of the object in the array: [reference]
``` python
arr.dtype
```
dtype('int32')
```python
arr2 = np.array([1.2, 3.4, 5.6])
arr2.dtype
```
dtype('float64')
​
## Conclusion

NumPy's efficient array operations, mathematical functions, and integration capabilities make it a fundamental tool for numerical computing in Python. Its versatility, performance, and ease of use make it an indispensable component of the Python scientific computing ecosystem. Whether you're performing basic numerical computations or tackling complex machine learning tasks, NumPy provides the tools you need to efficiently manipulate and analyze numerical data.
