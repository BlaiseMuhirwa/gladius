
## Gladius

[![GitHub issues](https://img.shields.io/github/issues/BlaiseMuhirwa/gladius)](https://github.com/BlaiseMuhirwa/gladius/issues)
[![GitHub forks](https://img.shields.io/github/forks/BlaiseMuhirwa/gladius)](https://github.com/BlaiseMuhirwa/gladius/network)
[![GitHub stars](https://img.shields.io/github/stars/BlaiseMuhirwa/gladius)](https://github.com/BlaiseMuhirwa/gladius/stargazers)
[![GitHub license](https://img.shields.io/github/license/BlaiseMuhirwa/gladius)](https://github.com/BlaiseMuhirwa/gladius/blob/main/LICENSE)

I started this project to get a better understanding of dynamic computation graphs in [PyTorch](https://pytorch.org/), a framework 
I use and love. Inspired by [the original paper of DyNet](https://arxiv.org/pdf/1701.03980.pdf), this repository implements a light-weight general-purpose deep learning 
library from scratch. The ultimate goal is to have a well-maintained C++ library that can be used to end-to-end training. 

### Architectural Design
Fortis's architecture consists of the following components: 

- `Parameter`: real-valued vectors and matrices representing weight matrices
        and bias vectors. 
- `LookupParameters`: sets of vectors of `Parameter`. 
- `Model`: Collection of `Parameter` and `LookupParameter` objects. The model keeps track 
        of the parameters and their gradients. 
- `Trainer`: implements an online update rule, such as SGD or Adam. The trainer holds a 
        pointer to the model object and, therefore, the parameters it contains. 
- `Expression`: Main data type being manipulated in a Fortis program. Each expression
        represents a subcomputation in the computation graph. For instance, a `Parameter` 
        object can be added to the computation graph, resulting in an expression W or b. 

- `Operations`: These are functions that act on expressions and return other expressions. 
        Crucially, they are not objects. Fortis defines many different operations, including
        addition, multiplication, softmax, tanh, etc. 
- `Builder classes`: These define interfaces for building different networks. In our case, we
        will mostly be interested in implementing the transformer network, but one should not 
        have a hard time having a recurrent neural network builder, for instance. 
        These work on top of expressions and operations and provide easy-to-use libraries. 
        More discussion on builders below. 
- `ComputationGraph`: Expressions are part of an implicit computation graph object, internally represented as a Directed Acyclic Graph. 
        Fortis currently assumes that only one computation graph will exist at a time. 
        From the user's perspective, we create a computation graph for each new training 
        example.
        
### Getting Started
Before cloning the repository, make sure you have the following installed on your
machine:
- cmake (version >= 3.18)
- cmake-format 
- clang-format 

While cloning this repository remember to also grab the submodules since we are using the 
following submodule dependencies: [cereal](https://uscilab.github.io/cereal/), [googletest](http://google.github.io/googletest/),
and [benchmark](https://github.com/google/benchmark). Then, build the library as follows (this will also build the unit tests):

```shell
$ git clone https://github.com/BlaiseMuhirwa/fortis.git --recurse-submodules
```
To build all unit and integration tests, you can pass an optional `tests` argument as follows

```shell
$ ./build.sh tests
```

If you do not provide the `tests` argument, `cmake` will only build Fortis static library. 

Note: We currently only support Macs with x86-64 architectures. Support for more architectures will be added progressively. 

### Upcoming Features and Optimization
- Full implementation of the Transformer architecture (scaled dot-product attention, etc.). This will be fun.
- Fast computations with [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
- Support for data-parallel training 
- SGD and Adam 
- Parallel implementations with OpenMP. 
- No more Jacobian computations. Nobody computes the Jacobian, but I figured it is mathematically helpful for my understanding to actually 
derive every single gradient update. This turns out to be very inefficient since the Jacobian for Fully-Connected Layers is very sparse, so we 
can benefit a lot from not computing it. 

To get a sense of the sparsity of the Jacobian, consider the following case. Suppose we have a weight matrix $W \in \mathbb{R}^{m\times n}$ and a vector of activations computed by the ReLU function, $z \in \mathbb{R}^{n}$. Let $\Phi: \mathbb{R}^{m\times n}\times \mathbb{R}^{n} \to \mathbb{R}^{m}$ be the map

$$
\Phi(W,z) = Wz
$$

Let $D_{W}\Phi \in \mathbb{R}^{m \times (m\times n)}$ be the Jacobian of the map w.r.t the weight matrix. You can convince yourself that it is given by

$$
\begin{bmatrix}
z_1 & z_2 & \ldots & z_n & \ldots & 0 & 0 & \ldots & 0 \\
\vdots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \vdots \\
0 & 0 & \ldots & 0 & \ldots & z_1 & z_2 & \ldots & z_n
\end{bmatrix}
$$

Notice that only $\frac{1}{m}$ entries all non-zero. So, for large values of $m$, $D_{W}\Phi$ is very sparse and we are far better off not computing the matrix above. 

